// Function: sub_26EFC50
// Address: 0x26efc50
//
void __fastcall sub_26EFC50(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  char v11; // al
  __int64 *v12; // rdx
  __int64 *v13; // r12
  __int64 *v14; // rbx
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  int v20; // [rsp+18h] [rbp-48h]
  char v21; // [rsp+1Ch] [rbp-44h]
  char v22; // [rsp+20h] [rbp-40h] BYREF

  v18 = (__int64 *)&v22;
  v6 = *(_DWORD *)(a1 + 4);
  v17 = 0;
  v19 = 4;
  v20 = 0;
  v7 = 4LL * (v6 & 0x7FFFFFF);
  v21 = 1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v8 = *(__int64 **)(a1 - 8);
    v9 = (__int64)&v8[v7];
  }
  else
  {
    v9 = a1;
    v8 = (__int64 *)(a1 - v7 * 8);
  }
  if ( v8 != (__int64 *)v9 )
  {
    while ( 1 )
    {
      a2 = *v8;
      v10 = *(_QWORD *)(*v8 + 16);
      if ( v10 )
        break;
LABEL_18:
      if ( v21 )
      {
        v15 = v18;
        a4 = HIDWORD(v19);
        a3 = &v18[HIDWORD(v19)];
        if ( v18 == a3 )
        {
LABEL_24:
          if ( HIDWORD(v19) >= (unsigned int)v19 )
            goto LABEL_25;
          a4 = (unsigned int)(HIDWORD(v19) + 1);
          v8 += 4;
          ++HIDWORD(v19);
          *a3 = a2;
          ++v17;
          if ( (__int64 *)v9 == v8 )
            goto LABEL_9;
        }
        else
        {
          while ( a2 != *v15 )
          {
            if ( a3 == ++v15 )
              goto LABEL_24;
          }
LABEL_8:
          v8 += 4;
          if ( (__int64 *)v9 == v8 )
            goto LABEL_9;
        }
      }
      else
      {
LABEL_25:
        v8 += 4;
        sub_C8CC70((__int64)&v17, a2, (__int64)a3, a4, a5, a6);
        if ( (__int64 *)v9 == v8 )
          goto LABEL_9;
      }
    }
    while ( *(_QWORD *)(v10 + 24) == a1 )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_18;
    }
    goto LABEL_8;
  }
LABEL_9:
  if ( *(_BYTE *)a1 == 3 )
  {
    if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    {
      if ( v21 )
        return;
LABEL_30:
      _libc_free((unsigned __int64)v18);
      return;
    }
    sub_B30290(a1);
    goto LABEL_12;
  }
  if ( !*(_BYTE *)a1 || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) - 15) > 3u )
  {
LABEL_12:
    v11 = v21;
    v12 = v18;
    if ( v21 )
      goto LABEL_13;
    goto LABEL_42;
  }
  sub_ACFDF0((__int64 *)a1, a2, (__int64)a3);
  v11 = v21;
  v12 = v18;
  if ( v21 )
  {
LABEL_13:
    v13 = &v12[HIDWORD(v19)];
    if ( v12 != v13 )
      goto LABEL_14;
    return;
  }
LABEL_42:
  v13 = &v12[(unsigned int)v19];
  if ( v12 == v13 )
    goto LABEL_30;
LABEL_14:
  while ( 1 )
  {
    v14 = v12;
    if ( (unsigned __int64)*v12 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v13 == ++v12 )
      goto LABEL_16;
  }
  if ( v13 != v12 )
  {
    do
    {
      sub_26EFC50();
      v16 = v14 + 1;
      if ( v14 + 1 == v13 )
        break;
      while ( 1 )
      {
        v14 = v16;
        if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v13 == ++v16 )
          goto LABEL_35;
      }
    }
    while ( v16 != v13 );
LABEL_35:
    v11 = v21;
  }
LABEL_16:
  if ( !v11 )
    goto LABEL_30;
}
