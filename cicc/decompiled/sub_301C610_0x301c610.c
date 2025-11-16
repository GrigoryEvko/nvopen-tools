// Function: sub_301C610
// Address: 0x301c610
//
_QWORD *__fastcall sub_301C610(_QWORD *a1, char *a2, __int64 a3)
{
  char v4; // al
  char v5; // al
  _BYTE *v6; // r14
  char v7; // bl
  _BYTE *v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r15
  _QWORD *v16; // rsi
  _QWORD *v17; // rdx
  _BYTE v19[4]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v20; // [rsp+4h] [rbp-8Ch]
  __int64 v21; // [rsp+Ch] [rbp-84h]
  __int128 v22; // [rsp+14h] [rbp-7Ch]
  __int64 v23; // [rsp+24h] [rbp-6Ch]
  __int64 v24; // [rsp+30h] [rbp-60h]
  __int64 v25; // [rsp+38h] [rbp-58h]
  __int64 v26; // [rsp+40h] [rbp-50h]
  unsigned int v27; // [rsp+48h] [rbp-48h]
  _BYTE *v28; // [rsp+50h] [rbp-40h]
  __int64 v29; // [rsp+58h] [rbp-38h]
  _BYTE v30[48]; // [rsp+60h] [rbp-30h] BYREF

  v4 = *a2;
  v20 = 0;
  v21 = 0;
  v19[0] = v4;
  v22 = 0u;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = v30;
  v29 = 0;
  v5 = sub_301BCE0((__int64)v19, a3);
  v6 = v28;
  v7 = v5;
  v8 = &v28[32 * (unsigned int)v29];
  if ( v28 != v8 )
  {
    do
    {
      v9 = *((_QWORD *)v8 - 3);
      v8 -= 32;
      if ( v9 )
        j_j___libc_free_0(v9);
    }
    while ( v6 != v8 );
    v8 = v28;
  }
  if ( v8 != v30 )
    _libc_free((unsigned __int64)v8);
  sub_C7D6A0(v25, 16LL * v27, 8);
  v10 = HIDWORD(v23);
  if ( HIDWORD(v23) )
  {
    v11 = *(_QWORD **)((char *)&v22 + 4);
    v12 = *(_QWORD *)((char *)&v22 + 4) + 16LL * HIDWORD(v23);
    do
    {
      if ( *v11 != -8192 && *v11 != -4096 )
      {
        v13 = v11[1];
        if ( v13 )
        {
          if ( (v13 & 4) != 0 )
          {
            v14 = (unsigned __int64 *)(v13 & 0xFFFFFFFFFFFFFFF8LL);
            v15 = (unsigned __int64)v14;
            if ( v14 )
            {
              if ( (unsigned __int64 *)*v14 != v14 + 2 )
                _libc_free(*v14);
              j_j___libc_free_0(v15);
            }
          }
        }
      }
      v11 += 2;
    }
    while ( (_QWORD *)v12 != v11 );
    v10 = HIDWORD(v23);
  }
  sub_C7D6A0(*(__int64 *)((char *)&v22 + 4), 16 * v10, 8);
  v16 = a1 + 4;
  v17 = a1 + 10;
  if ( v7 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v16;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v17;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v16;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v17;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
