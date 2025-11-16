// Function: sub_27A32A0
// Address: 0x27a32a0
//
void __fastcall sub_27A32A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  char v9; // di
  _BYTE *v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // r14
  __int64 v14; // r13
  __int64 *v15; // r15
  __int64 v16; // rdx
  char *v17; // rax
  __int64 v18; // rcx
  char *v19; // rsi
  __int64 v20; // rdx
  char *v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  int v29; // [rsp+18h] [rbp-58h]
  char v30; // [rsp+1Ch] [rbp-54h]
  char v31; // [rsp+20h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 + 16);
  v27 = (__int64 *)&v31;
  v26 = 0;
  v28 = 4;
  v29 = 0;
  v30 = 1;
  if ( !v6 )
    goto LABEL_15;
  v9 = 1;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = *(_BYTE **)(v6 + 24);
        if ( *v10 == 28 )
          break;
LABEL_9:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_10;
      }
      if ( v9 )
        break;
LABEL_17:
      sub_C8CC70((__int64)&v26, (__int64)v10, (__int64)a3, a4, a5, a6);
      v6 = *(_QWORD *)(v6 + 8);
      v9 = v30;
      if ( !v6 )
        goto LABEL_10;
    }
    v11 = v27;
    a4 = HIDWORD(v28);
    a3 = &v27[HIDWORD(v28)];
    if ( v27 != a3 )
    {
      while ( v10 != (_BYTE *)*v11 )
      {
        if ( a3 == ++v11 )
          goto LABEL_19;
      }
      goto LABEL_9;
    }
LABEL_19:
    if ( HIDWORD(v28) >= (unsigned int)v28 )
      goto LABEL_17;
    a4 = (unsigned int)++HIDWORD(v28);
    *a3 = (__int64)v10;
    v6 = *(_QWORD *)(v6 + 8);
    ++v26;
    v9 = v30;
  }
  while ( v6 );
LABEL_10:
  v12 = v27;
  if ( v9 )
    v13 = &v27[HIDWORD(v28)];
  else
    v13 = &v27[(unsigned int)v28];
  if ( v27 != v13 )
  {
    while ( 1 )
    {
      v14 = *v12;
      v15 = v12;
      if ( (unsigned __int64)*v12 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v13 == ++v12 )
        goto LABEL_15;
    }
    if ( v12 != v13 )
    {
      do
      {
        v16 = 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
        {
          v17 = *(char **)(v14 - 8);
          v18 = v16 >> 5;
          v19 = &v17[v16];
          v20 = v16 >> 7;
          if ( v20 )
            goto LABEL_25;
        }
        else
        {
          v19 = (char *)v14;
          v17 = (char *)(v14 - v16);
          v18 = v16 >> 5;
          v20 = v16 >> 7;
          if ( v20 )
          {
LABEL_25:
            v21 = &v17[128 * v20];
            while ( *(_QWORD *)v17 == a2 )
            {
              if ( *((_QWORD *)v17 + 4) != a2 )
              {
                v17 += 32;
                break;
              }
              if ( *((_QWORD *)v17 + 8) != a2 )
              {
                v17 += 64;
                break;
              }
              if ( *((_QWORD *)v17 + 12) != a2 )
              {
                v17 += 96;
                break;
              }
              v17 += 128;
              if ( v17 == v21 )
              {
                v18 = (v19 - v17) >> 5;
                goto LABEL_39;
              }
            }
LABEL_31:
            if ( v19 != v17 )
              goto LABEL_32;
            goto LABEL_42;
          }
        }
LABEL_39:
        if ( v18 != 2 )
        {
          if ( v18 != 3 )
          {
            if ( v18 != 1 )
              goto LABEL_42;
            goto LABEL_47;
          }
          if ( *(_QWORD *)v17 != a2 )
            goto LABEL_31;
          v17 += 32;
        }
        if ( *(_QWORD *)v17 != a2 )
          goto LABEL_31;
        v17 += 32;
LABEL_47:
        if ( *(_QWORD *)v17 != a2 )
          goto LABEL_31;
LABEL_42:
        sub_BD84D0(v14, a2);
        sub_D6E4B0(*(_QWORD **)(a1 + 256), v14, 0, v23, v24, v25);
LABEL_32:
        v22 = v15 + 1;
        if ( v15 + 1 == v13 )
          break;
        v14 = *v22;
        for ( ++v15; (unsigned __int64)*v22 >= 0xFFFFFFFFFFFFFFFELL; v15 = v22 )
        {
          if ( v13 == ++v22 )
            goto LABEL_15;
          v14 = *v22;
        }
      }
      while ( v13 != v15 );
    }
  }
LABEL_15:
  if ( !v30 )
    _libc_free((unsigned __int64)v27);
}
