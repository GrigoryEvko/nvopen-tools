// Function: sub_18B2C00
// Address: 0x18b2c00
//
void __fastcall sub_18B2C00(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rax
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // r15
  char v7; // al
  __int64 *v8; // r8
  __int64 *v9; // rdx
  __int64 *v10; // rbx
  __int64 *v11; // rax
  __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 *v14; // rax
  __int64 *v15; // rsi
  __int64 *v16; // rcx
  __int64 *v17; // rax
  int v18; // eax
  __int64 v19; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-78h]
  __int64 *v21; // [rsp+20h] [rbp-70h]
  __int64 v22; // [rsp+28h] [rbp-68h]
  int v23; // [rsp+30h] [rbp-60h]
  _BYTE v24[88]; // [rsp+38h] [rbp-58h] BYREF

  v20 = (__int64 *)v24;
  v21 = (__int64 *)v24;
  v1 = *(_DWORD *)(a1 + 20);
  v19 = 0;
  v22 = 4;
  v23 = 0;
  v2 = 3LL * (v1 & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v3 = *(__int64 **)(a1 - 8);
    v4 = (__int64)&v3[v2];
  }
  else
  {
    v4 = a1;
    v3 = (__int64 *)(a1 - v2 * 8);
  }
  if ( (__int64 *)v4 != v3 )
  {
LABEL_4:
    while ( 1 )
    {
      v5 = *v3;
      v6 = *(_QWORD *)(*v3 + 8);
      if ( v6 )
        break;
LABEL_22:
      v14 = v20;
      if ( v21 == v20 )
      {
        v15 = &v20[HIDWORD(v22)];
        if ( v20 == v15 )
        {
LABEL_50:
          if ( HIDWORD(v22) >= (unsigned int)v22 )
            goto LABEL_23;
          ++HIDWORD(v22);
          *v15 = v5;
          ++v19;
        }
        else
        {
          v16 = 0;
          while ( v5 != *v14 )
          {
            if ( *v14 == -2 )
              v16 = v14;
            if ( v15 == ++v14 )
            {
              if ( !v16 )
                goto LABEL_50;
              v3 += 3;
              *v16 = v5;
              --v23;
              ++v19;
              if ( (__int64 *)v4 != v3 )
                goto LABEL_4;
              goto LABEL_9;
            }
          }
        }
LABEL_8:
        v3 += 3;
        if ( (__int64 *)v4 == v3 )
          goto LABEL_9;
      }
      else
      {
LABEL_23:
        v3 += 3;
        sub_16CCBA0((__int64)&v19, v5);
        if ( (__int64 *)v4 == v3 )
          goto LABEL_9;
      }
    }
    while ( (_QWORD *)a1 == sub_1648700(v6) )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        goto LABEL_22;
    }
    goto LABEL_8;
  }
LABEL_9:
  v7 = *(_BYTE *)(a1 + 16);
  if ( v7 == 3 )
  {
    if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    {
      v13 = (unsigned __int64)v21;
      if ( v20 != v21 )
        goto LABEL_20;
      return;
    }
    sub_15E55B0(a1);
  }
  else if ( v7 )
  {
    v18 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(v18 - 13) <= 1 || v18 == 16 )
    {
      sub_159D850(a1);
      v8 = v21;
      v9 = v20;
      if ( v21 != v20 )
        goto LABEL_13;
      goto LABEL_49;
    }
  }
  v8 = v21;
  v9 = v20;
  if ( v21 != v20 )
  {
LABEL_13:
    v10 = &v8[(unsigned int)v22];
    goto LABEL_14;
  }
LABEL_49:
  v10 = &v8[HIDWORD(v22)];
LABEL_14:
  if ( v8 != v10 )
  {
    v11 = v8;
    while ( 1 )
    {
      v12 = v11;
      if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v11 )
        goto LABEL_18;
    }
    if ( v11 != v10 )
    {
      do
      {
        sub_18B2C00();
        v17 = v12 + 1;
        if ( v12 + 1 == v10 )
          break;
        while ( 1 )
        {
          v12 = v17;
          if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v17 )
            goto LABEL_41;
        }
      }
      while ( v10 != v17 );
LABEL_41:
      v8 = v21;
      v9 = v20;
    }
  }
LABEL_18:
  if ( v8 != v9 )
  {
    v13 = (unsigned __int64)v8;
LABEL_20:
    _libc_free(v13);
  }
}
