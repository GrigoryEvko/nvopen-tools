// Function: sub_2A7F790
// Address: 0x2a7f790
//
__int64 __fastcall sub_2A7F790(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  void *v5; // rsi
  __int64 v6; // r13
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 i; // r14
  __int64 *v11; // rbx
  __int64 *v12; // r14
  int v13; // r15d
  _QWORD *v14; // r13
  __int64 v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // rdx
  int v22; // eax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // [rsp+18h] [rbp-118h]
  _QWORD *v28; // [rsp+18h] [rbp-118h]
  char *v29; // [rsp+20h] [rbp-110h] BYREF
  char v30; // [rsp+40h] [rbp-F0h]
  char v31; // [rsp+41h] [rbp-EFh]
  __int64 v32; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v33; // [rsp+58h] [rbp-D8h]
  _DWORD v34[3]; // [rsp+60h] [rbp-D0h] BYREF
  char v35; // [rsp+6Ch] [rbp-C4h]
  _QWORD v36[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+80h] [rbp-B0h] BYREF
  _BYTE *v38; // [rsp+88h] [rbp-A8h]
  __int64 v39; // [rsp+90h] [rbp-A0h]
  int v40; // [rsp+98h] [rbp-98h]
  char v41; // [rsp+9Ch] [rbp-94h]
  _BYTE v42[144]; // [rsp+A0h] [rbp-90h] BYREF

  if ( sub_B2FC80(a3) )
  {
    v5 = (void *)(a1 + 32);
    v6 = a1 + 80;
LABEL_3:
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v6;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v8 = a3 + 72;
  v9 = *(_QWORD *)(a3 + 80);
  v32 = (__int64)v34;
  v33 = 0x1400000000LL;
  if ( v8 == v9 )
  {
    i = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v9 + 32);
      if ( i != v9 + 24 )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v8 == v9 )
        goto LABEL_11;
      if ( !v9 )
        BUG();
    }
  }
  v18 = 0x8000000000041LL;
  while ( v8 != v9 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 85 )
    {
      v20 = *(_QWORD *)(i - 56);
      if ( v20 )
      {
        if ( !*(_BYTE *)v20
          && *(_QWORD *)(v20 + 24) == *(_QWORD *)(i + 56)
          && (*(_BYTE *)(v20 + 33) & 0x20) != 0
          && *(_DWORD *)(v20 + 36) == 149 )
        {
          v21 = *(unsigned __int8 **)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24);
          v22 = *v21;
          if ( (unsigned __int8)v22 > 0x1Cu )
          {
            v23 = (unsigned int)(v22 - 34);
            if ( (unsigned __int8)v23 <= 0x33u )
            {
              if ( _bittest64(&v18, v23) )
              {
                v24 = *((_QWORD *)v21 - 4);
                if ( v24 )
                {
                  if ( !*(_BYTE *)v24 && *(_QWORD *)(v24 + 24) == *((_QWORD *)v21 + 10) && *(_DWORD *)(v24 + 36) == 151 )
                  {
                    v25 = (unsigned int)v33;
                    v26 = (unsigned int)v33 + 1LL;
                    if ( v26 > HIDWORD(v33) )
                    {
                      sub_C8D5F0((__int64)&v32, v34, v26, 8u, 0x8000000000041LL, v4);
                      v25 = (unsigned int)v33;
                      v18 = 0x8000000000041LL;
                    }
                    *(_QWORD *)(v32 + 8 * v25) = i - 24;
                    LODWORD(v33) = v33 + 1;
                  }
                }
              }
            }
          }
        }
      }
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v9 + 32) )
    {
      v19 = v9 - 24;
      if ( !v9 )
        v19 = 0;
      if ( i != v19 + 48 )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v8 == v9 )
        goto LABEL_11;
      if ( !v9 )
        BUG();
    }
  }
LABEL_11:
  v11 = (__int64 *)v32;
  v12 = (__int64 *)(v32 + 8LL * (unsigned int)v33);
  v13 = v33;
  if ( (__int64 *)v32 != v12 )
  {
    do
    {
      v14 = (_QWORD *)*v11;
      v15 = sub_B5B890(*v11);
      v27 = v14[1];
      if ( v27 != *(_QWORD *)(v15 + 8) )
      {
        v31 = 1;
        v29 = "cast";
        v30 = 3;
        v16 = sub_BD2C40(72, 1u);
        if ( v16 )
        {
          v17 = v27;
          v28 = v16;
          sub_B51BF0((__int64)v16, v15, v17, (__int64)&v29, (__int64)(v14 + 3), 0);
          v16 = v28;
        }
        v15 = (__int64)v16;
      }
      ++v11;
      sub_BD84D0((__int64)v14, v15);
      sub_B43D60(v14);
    }
    while ( v12 != v11 );
    v13 = v33;
    v12 = (__int64 *)v32;
  }
  if ( v12 != (__int64 *)v34 )
    _libc_free((unsigned __int64)v12);
  v5 = (void *)(a1 + 32);
  v6 = a1 + 80;
  if ( !v13 )
    goto LABEL_3;
  v33 = (unsigned __int64)v36;
  v36[0] = &unk_4F82408;
  v34[0] = 2;
  v34[2] = 0;
  v35 = 1;
  v37 = 0;
  v38 = v42;
  v39 = 2;
  v40 = 0;
  v41 = 1;
  v34[1] = 1;
  v32 = 1;
  sub_C8CF70(a1, v5, 2, (__int64)v36, (__int64)&v32);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v42, (__int64)&v37);
  if ( !v41 )
    _libc_free((unsigned __int64)v38);
  if ( !v35 )
    _libc_free(v33);
  return a1;
}
