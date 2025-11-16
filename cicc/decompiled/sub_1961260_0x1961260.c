// Function: sub_1961260
// Address: 0x1961260
//
__int64 __fastcall sub_1961260(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        char *a16,
        __int64 *a17,
        unsigned __int8 a18)
{
  unsigned int v19; // r13d
  __int64 **v20; // r14
  int v21; // edx
  __int64 v22; // rcx
  __int64 v23; // rax
  int v24; // edi
  __int64 v25; // rsi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  __int64 v29; // rbx
  int v30; // r14d
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // r13
  _QWORD *v34; // r12
  __int64 v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  _QWORD *v41; // rax
  bool v42; // r9
  _QWORD *v43; // rax
  __int64 *v44; // r14
  __int64 *v45; // rax
  __int64 v46; // r14
  int v47; // eax
  __int64 **v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r13
  int v51; // eax
  double v52; // xmm4_8
  double v53; // xmm5_8
  int v54; // eax
  int v55; // r9d
  __int64 **v57; // [rsp+10h] [rbp-120h]
  char *v60; // [rsp+28h] [rbp-108h]
  bool v61; // [rsp+35h] [rbp-FBh]
  char v62; // [rsp+37h] [rbp-F9h]
  __int64 v63; // [rsp+38h] [rbp-F8h]
  __int64 v64; // [rsp+40h] [rbp-F0h]
  _BYTE v66[16]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v67; // [rsp+60h] [rbp-D0h]
  char *v68; // [rsp+70h] [rbp-C0h] BYREF
  int v69; // [rsp+78h] [rbp-B8h]
  char v70; // [rsp+80h] [rbp-B0h] BYREF

  v19 = 0;
  sub_1B1AE10(&v68, a1, a6);
  v60 = &v68[8 * v69];
  if ( v68 != v60 )
  {
    v20 = (__int64 **)v68;
    while ( 1 )
    {
      while ( 1 )
      {
        v21 = *(_DWORD *)(a3 + 24);
        v22 = **v20;
        v23 = 0;
        if ( v21 )
        {
          v24 = v21 - 1;
          v25 = *(_QWORD *)(a3 + 8);
          v26 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v22 == *v27 )
          {
LABEL_6:
            v23 = v27[1];
          }
          else
          {
            v54 = 1;
            while ( v28 != -8 )
            {
              v55 = v54 + 1;
              v26 = v24 & (v54 + v26);
              v27 = (__int64 *)(v25 + 16LL * v26);
              v28 = *v27;
              if ( v22 == *v27 )
                goto LABEL_6;
              v54 = v55;
            }
            v23 = 0;
          }
        }
        if ( a6 == v23 )
        {
          v29 = *(_QWORD *)(v22 + 48);
          v64 = v22 + 40;
          v62 = **(_QWORD **)(a6 + 32) == v22;
          if ( v22 + 40 != v29 )
            break;
        }
        if ( v60 == (char *)++v20 )
          goto LABEL_19;
      }
      v57 = v20;
      v30 = v19;
      do
      {
        while ( 1 )
        {
          v33 = v29;
          v29 = *(_QWORD *)(v29 + 8);
          v34 = (_QWORD *)(v33 - 24);
          v35 = sub_15F2050(v33 - 24);
          v36 = (_BYTE *)sub_1632FA0(v35);
          v37 = sub_14DD210((__int64 *)(v33 - 24), v36, a5);
          if ( v37 )
          {
            v63 = v37;
            sub_135BB20(a15, v33 - 24, v37);
            sub_164D160(v33 - 24, v63, a7, a8, a9, a10, v31, v32, a13, a14);
            v30 = sub_1AE9990(v33 - 24, a5);
            if ( (_BYTE)v30 )
            {
              sub_1359860(a15, v33 - 24);
              sub_15F20C0((_QWORD *)(v33 - 24));
            }
            else
            {
              v30 = 1;
            }
            goto LABEL_12;
          }
          if ( sub_13FC1D0(a6, v33 - 24) && (unsigned __int8)sub_1960590(v33 - 24, a2, a4, a6, a15, a16, a17) )
          {
            if ( v62
              || (v39 = sub_13FC520(a6), v40 = sub_157EBA0(v39), (unsigned __int8)sub_14AF470(v33 - 24, v40, a4, a18))
              || (unsigned __int8)sub_195F310(v33 - 24, a4, a6, a16, a17) )
            {
              v30 |= sub_1961180(v33 - 24, a4, a6, a16, a17);
              goto LABEL_12;
            }
            if ( *(_BYTE *)(v33 - 8) != 43 )
              goto LABEL_12;
          }
          else if ( *(_BYTE *)(v33 - 8) != 43 )
          {
            goto LABEL_16;
          }
          if ( (*(_BYTE *)(v33 - 1) & 0x40) != 0 )
            v41 = *(_QWORD **)(v33 - 32);
          else
            v41 = &v34[-3 * (*(_DWORD *)(v33 - 4) & 0xFFFFFFF)];
          if ( sub_13FC1A0(a6, v41[3]) )
          {
            v42 = sub_15F24D0(v33 - 24);
            if ( v42 )
            {
              if ( (*(_BYTE *)(v33 - 1) & 0x40) != 0 )
                v43 = *(_QWORD **)(v33 - 32);
              else
                v43 = &v34[-3 * (*(_DWORD *)(v33 - 4) & 0xFFFFFFF)];
              v44 = (__int64 *)v43[3];
              v61 = v42;
              a7 = (__m128)0x3FF0000000000000uLL;
              v45 = (__int64 *)sub_15A10B0(*v44, 1.0);
              v67 = 257;
              v46 = sub_15FB440(19, v45, (__int64)v44, (__int64)v66, 0);
              v47 = sub_15F24E0(v33 - 24);
              sub_15F2440(v46, v47);
              sub_15F2120(v46, v33 - 24);
              v67 = 257;
              if ( (*(_BYTE *)(v33 - 1) & 0x40) != 0 )
                v48 = *(__int64 ***)(v33 - 32);
              else
                v48 = (__int64 **)&v34[-3 * (*(_DWORD *)(v33 - 4) & 0xFFFFFFF)];
              v49 = v33 - 24;
              v50 = sub_15FB440(16, *v48, v46, (__int64)v66, 0);
              v51 = sub_15F24E0(v49);
              sub_15F2440(v50, v51);
              sub_15F2180(v50, (__int64)v34);
              sub_164D160((__int64)v34, v50, (__m128)0x3FF0000000000000uLL, a8, a9, a10, v52, v53, a13, a14);
              sub_15F20C0(v34);
              sub_1961180(v46, a4, a6, a16, a17);
              v30 = v61;
              goto LABEL_12;
            }
          }
LABEL_16:
          if ( v62 )
            break;
LABEL_12:
          if ( v64 == v29 )
            goto LABEL_18;
        }
        v62 = sub_14AE440(v33 - 24);
      }
      while ( v64 != v29 );
LABEL_18:
      v19 = v30;
      v20 = v57 + 1;
      if ( v60 == (char *)(v57 + 1) )
      {
LABEL_19:
        v60 = v68;
        break;
      }
    }
  }
  if ( v60 != &v70 )
    _libc_free((unsigned __int64)v60);
  return v19;
}
