// Function: sub_1AA2F10
// Address: 0x1aa2f10
//
__int64 __fastcall sub_1AA2F10(
        __m128 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        __m128 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 v15; // rbx
  __int64 v16; // r9
  __int64 v17; // r12
  __int64 i; // r15
  unsigned __int8 v19; // al
  __int64 v20; // r13
  __int64 v21; // rdi
  __int64 v22; // rax
  int v23; // r9d
  unsigned __int8 v24; // al
  unsigned __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 v29; // r14
  char v30; // r12
  __int64 v31; // rdi
  __int64 j; // r14
  __int64 v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v37; // rdx
  unsigned __int64 v41; // [rsp+30h] [rbp-3D0h]
  unsigned __int8 v42; // [rsp+3Fh] [rbp-3C1h]
  __int64 v43; // [rsp+40h] [rbp-3C0h] BYREF
  _BYTE *v44; // [rsp+48h] [rbp-3B8h]
  __int64 v45; // [rsp+50h] [rbp-3B0h]
  _BYTE v46[256]; // [rsp+58h] [rbp-3A8h] BYREF
  __int64 v47; // [rsp+158h] [rbp-2A8h]
  _BYTE *v48; // [rsp+160h] [rbp-2A0h]
  _BYTE *v49; // [rsp+168h] [rbp-298h]
  __int64 v50; // [rsp+170h] [rbp-290h]
  int v51; // [rsp+178h] [rbp-288h]
  _BYTE v52[64]; // [rsp+180h] [rbp-280h] BYREF
  _BYTE *v53; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 v54; // [rsp+1C8h] [rbp-238h]
  _BYTE v55[560]; // [rsp+1D0h] [rbp-230h] BYREF

  v44 = v46;
  v43 = a11;
  v45 = 0x1000000000LL;
  v47 = 0;
  v48 = v52;
  v49 = v52;
  v50 = 8;
  v51 = 0;
  v42 = sub_1AF0CE0(a10, 0, &v43);
  sub_15DC150((__int64)&v43);
  v15 = *(_QWORD *)(a10 + 80);
  v53 = v55;
  v54 = 0x4000000000LL;
  if ( a10 + 72 == v15 )
  {
    v16 = 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    while ( 1 )
    {
      v16 = *(_QWORD *)(v15 + 24);
      if ( v16 != v15 + 16 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( a10 + 72 == v15 )
        break;
      if ( !v15 )
        BUG();
    }
  }
  v17 = a10 + 72;
  i = v16;
  while ( v15 != v17 )
  {
    if ( !i )
      BUG();
    v19 = *(_BYTE *)(i - 8);
    v20 = i - 24;
    if ( v19 > 0x17u )
    {
      v21 = v20 | 4;
      if ( v19 != 78 )
      {
        if ( v19 != 29 )
          goto LABEL_13;
        v21 = v20 & 0xFFFFFFFFFFFFFFFBLL;
      }
      if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0 && !(unsigned __int8)sub_1AEC650(v21, a13) && !sub_1642CF0(v21) )
      {
        v24 = *(_BYTE *)(i - 8);
        v25 = 0;
        if ( v24 > 0x17u )
        {
          v25 = v20 | 4;
          if ( v24 != 78 )
          {
            v25 = v20 & 0xFFFFFFFFFFFFFFFBLL;
            if ( v24 != 29 )
              v25 = 0;
          }
        }
        v26 = (unsigned int)v54;
        if ( (unsigned int)v54 >= HIDWORD(v54) )
        {
          v41 = v25;
          sub_16CD150((__int64)&v53, v55, 0, 8, v25, v23);
          v26 = (unsigned int)v54;
          v25 = v41;
        }
        *(_QWORD *)&v53[8 * v26] = v25;
        LODWORD(v54) = v54 + 1;
      }
    }
LABEL_13:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v15 + 24) )
    {
      v22 = v15 - 24;
      if ( !v15 )
        v22 = 0;
      if ( i != v22 + 40 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v17 == v15 )
        break;
      if ( !v15 )
        BUG();
    }
  }
  v27 = a10;
  v28 = a10 + 72;
  if ( (_DWORD)v54 )
  {
    v29 = *(_QWORD *)(a10 + 80);
    if ( v29 != v17 )
    {
      v30 = v42;
      do
      {
        v31 = v29 - 24;
        if ( !v29 )
          v31 = 0;
        if ( sub_157F120(v31) )
        {
          v30 = 1;
          sub_1AA62D0(v31, 0);
        }
        v29 = *(_QWORD *)(v29 + 8);
      }
      while ( v29 != v28 );
      v27 = a10;
      v42 = v30;
      for ( j = *(_QWORD *)(a10 + 80); j != v28; j = *(_QWORD *)(j + 8) )
      {
        v33 = j - 24;
        if ( !j )
          v33 = 0;
        v34 = sub_157EBA0(v33);
        if ( *(_BYTE *)(v34 + 16) == 26 && (*(_DWORD *)(v34 + 20) & 0xFFFFFFF) == 3 )
        {
          v35 = *(_QWORD *)(v34 - 72);
          if ( *(_BYTE *)(v35 + 16) == 75 )
          {
            v37 = *(_QWORD *)(v35 + 8);
            if ( v37 )
            {
              if ( !*(_QWORD *)(v37 + 8) )
              {
                sub_15F22F0((_QWORD *)v35, v34);
                v42 = 1;
              }
            }
          }
        }
      }
    }
    v42 |= sub_1AA0070(v27, a11, a12, (__int64)&v53, a1, a2, a3, a4, v13, v14, a7, a8);
  }
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( v49 != v48 )
    _libc_free((unsigned __int64)v49);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  return v42;
}
