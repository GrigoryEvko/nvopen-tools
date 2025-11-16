// Function: sub_1D8CA60
// Address: 0x1d8ca60
//
__int64 __fastcall sub_1D8CA60(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 i; // r12
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 *v19; // rbx
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  int v26; // r13d
  __int64 v27; // rax
  int v28; // ecx
  char v29; // al
  __int64 v30; // r11
  int v31; // ecx
  void *v32; // rsi
  double v33; // xmm4_8
  double v34; // xmm5_8
  char v36; // al
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v40; // [rsp+8h] [rbp-E8h]
  __int64 v41; // [rsp+10h] [rbp-E0h]
  __int64 v42; // [rsp+10h] [rbp-E0h]
  int v43; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v44; // [rsp+1Fh] [rbp-D1h]
  __int64 *v46; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v47; // [rsp+38h] [rbp-B8h] BYREF
  __int64 *v48; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+48h] [rbp-A8h]
  _BYTE v50[32]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v51[5]; // [rsp+70h] [rbp-80h] BYREF
  int v52; // [rsp+98h] [rbp-58h]
  __int64 v53; // [rsp+A0h] [rbp-50h]
  __int64 v54; // [rsp+A8h] [rbp-48h]

  v14 = a1 + 72;
  v15 = *(_QWORD *)(a1 + 80);
  v48 = (__int64 *)v50;
  v49 = 0x400000000LL;
  if ( a1 + 72 == v15 )
  {
    return 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v15 + 24);
      if ( i != v15 + 16 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v14 == v15 )
        return 0;
      if ( !v15 )
        BUG();
    }
    while ( v15 != v14 )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 8) == 78 )
      {
        v37 = *(_QWORD *)(i - 48);
        if ( !*(_BYTE *)(v37 + 16) && (*(_BYTE *)(v37 + 33) & 0x20) != 0 )
        {
          v38 = (unsigned int)v49;
          if ( (unsigned int)v49 >= HIDWORD(v49) )
          {
            sub_16CD150((__int64)&v48, v50, 0, 8, a13, a14);
            v38 = (unsigned int)v49;
          }
          v48[v38] = i - 24;
          LODWORD(v49) = v49 + 1;
        }
      }
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v15 + 24) )
      {
        v17 = v15 - 24;
        if ( !v15 )
          v17 = 0;
        if ( i != v17 + 40 )
          break;
        v15 = *(_QWORD *)(v15 + 8);
        if ( v14 == v15 )
          break;
        if ( !v15 )
          BUG();
      }
    }
    v18 = v48;
    v46 = &v48[(unsigned int)v49];
    if ( v46 == v48 )
    {
      v44 = 0;
    }
    else
    {
      v44 = 0;
      v19 = v48;
      do
      {
        v20 = *v19;
        v21 = sub_16498A0(*v19);
        v53 = 0;
        v54 = 0;
        v22 = *(unsigned __int8 **)(v20 + 48);
        v51[3] = v21;
        v52 = 0;
        v23 = *(_QWORD *)(v20 + 40);
        v51[0] = 0;
        v51[1] = v23;
        v51[4] = 0;
        v51[2] = v20 + 24;
        v47 = v22;
        if ( v22 )
        {
          sub_1623A60((__int64)&v47, (__int64)v22, 2);
          if ( v51[0] )
            sub_161E7C0((__int64)v51, v51[0]);
          v51[0] = (__int64)v47;
          if ( v47 )
            sub_1623210((__int64)&v47, v47, (__int64)v51);
        }
        v24 = *(_QWORD *)(v20 - 24);
        if ( *(_BYTE *)(v24 + 16) )
          BUG();
        v25 = *(_DWORD *)(v24 + 36);
        v26 = v25 - 83;
        switch ( v25 )
        {
          case 'S':
          case 'T':
          case 'V':
          case 'W':
          case 'Y':
          case 'Z':
          case '[':
          case '\\':
          case ']':
          case '^':
          case '_':
            v27 = (unsigned int)(v25 - 86);
            v28 = 0;
            if ( (unsigned int)v27 <= 8 )
              v28 = dword_42E9220[v27];
            v39 = v28;
            v41 = *(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
            v29 = sub_14A3AA0(a2);
            v30 = v41;
            v31 = v39;
            if ( !v29 )
              goto LABEL_42;
            goto LABEL_30;
          case 'U':
          case 'X':
            v43 = sub_15F24E0(v20);
            v40 = *(unsigned __int8 **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
            v42 = *(_QWORD *)(v20 + 24 * (1LL - (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)));
            v36 = sub_14A3AA0(a2);
            v30 = v42;
            if ( !v36 )
              goto LABEL_42;
            if ( v43 == -1 )
            {
              v31 = 0;
LABEL_30:
              v32 = sub_1B19ED0(
                      (__int64)v51,
                      v30,
                      *(_DWORD *)&asc_42E91E0[4 * v26],
                      v31,
                      0,
                      0,
                      *(double *)a3.m128_u64,
                      a4,
                      a5);
            }
            else
            {
              v32 = sub_1B19C30(
                      v51,
                      v40,
                      v42,
                      *(_DWORD *)&asc_42E91E0[4 * v26],
                      0,
                      *(double *)a3.m128_u64,
                      a4,
                      a5,
                      (__int64)v40,
                      0,
                      0);
            }
            sub_164D160(v20, (__int64)v32, a3, a4, a5, a6, v33, v34, a9, a10);
            sub_15F20C0((_QWORD *)v20);
            if ( v51[0] )
              sub_161E7C0((__int64)v51, v51[0]);
            v44 = 1;
            break;
          default:
LABEL_42:
            if ( v51[0] )
              sub_161E7C0((__int64)v51, v51[0]);
            break;
        }
        ++v19;
      }
      while ( v46 != v19 );
      v18 = v48;
    }
    if ( v18 != (__int64 *)v50 )
      _libc_free((unsigned __int64)v18);
  }
  return v44;
}
