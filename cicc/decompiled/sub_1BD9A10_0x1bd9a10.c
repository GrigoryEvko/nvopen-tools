// Function: sub_1BD9A10
// Address: 0x1bd9a10
//
__int64 __fastcall sub_1BD9A10(
        __m128i a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        __m128 a8,
        __int64 a9,
        __int64 *a10,
        __int64 a11,
        __int64 a12,
        unsigned int a13,
        int a14)
{
  __int64 *v15; // r13
  __int64 *v16; // r12
  unsigned int v18; // eax
  unsigned int v19; // r9d
  unsigned int v20; // ebx
  __int64 *v22; // r10
  _BYTE *v23; // r8
  int v24; // edx
  _BYTE *v25; // rax
  _QWORD *v26; // r13
  __int64 *v27; // r12
  unsigned int v28; // r15d
  __int64 *v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned int v32; // r14d
  unsigned int v33; // r15d
  __int64 *v34; // rbx
  _BYTE *v35; // rdx
  __int64 *v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // r13
  unsigned int v39; // r12d
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 *v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  _QWORD *v48; // r8
  int v49; // r9d
  int v50; // r10d
  __int64 v51; // rbx
  __int64 v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  char *v55; // r15
  char *v56; // rbx
  char *v57; // rdi
  _QWORD *v58; // [rsp+8h] [rbp-3D8h]
  unsigned __int8 v59; // [rsp+10h] [rbp-3D0h]
  unsigned int v60; // [rsp+10h] [rbp-3D0h]
  unsigned int v61; // [rsp+20h] [rbp-3C0h]
  __int64 v62; // [rsp+20h] [rbp-3C0h]
  __int64 *v63; // [rsp+28h] [rbp-3B8h]
  unsigned int v64; // [rsp+28h] [rbp-3B8h]
  __int64 v65; // [rsp+30h] [rbp-3B0h]
  __int64 v66; // [rsp+30h] [rbp-3B0h]
  unsigned int v67; // [rsp+38h] [rbp-3A8h]
  unsigned __int8 v68; // [rsp+38h] [rbp-3A8h]
  _QWORD v69[2]; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 v70; // [rsp+50h] [rbp-390h] BYREF
  __int64 *v71; // [rsp+60h] [rbp-380h]
  __int64 v72; // [rsp+70h] [rbp-370h] BYREF
  _QWORD v73[2]; // [rsp+A0h] [rbp-340h] BYREF
  __int64 v74; // [rsp+B0h] [rbp-330h] BYREF
  __int64 *v75; // [rsp+C0h] [rbp-320h]
  __int64 v76; // [rsp+D0h] [rbp-310h] BYREF
  _BYTE *v77; // [rsp+100h] [rbp-2E0h] BYREF
  __int64 v78; // [rsp+108h] [rbp-2D8h]
  _BYTE v79[192]; // [rsp+110h] [rbp-2D0h] BYREF
  _QWORD v80[11]; // [rsp+1D0h] [rbp-210h] BYREF
  char *v81; // [rsp+228h] [rbp-1B8h]
  unsigned int v82; // [rsp+230h] [rbp-1B0h]
  char v83; // [rsp+238h] [rbp-1A8h] BYREF

  v15 = a10;
  v16 = (__int64 *)a12;
  v18 = sub_1BBE2D0(a12, *a10, a11, a12, a13, a14);
  v19 = 0;
  if ( v18 )
  {
    v20 = a13 / v18;
    LOBYTE(v19) = v20 <= 1 || (v18 & (v18 - 1)) != 0;
    if ( (_BYTE)v19 )
    {
      return 0;
    }
    else
    {
      v22 = &a10[a11];
      v77 = v79;
      v78 = 0x800000000LL;
      v65 = (8 * a11) >> 3;
      if ( (unsigned __int64)(8 * a11) > 0x40 )
      {
        sub_170B450((__int64)&v77, (8 * a11) >> 3);
        v24 = v78;
        v23 = v77;
        v22 = &a10[a11];
        v19 = 0;
        v25 = &v77[24 * (unsigned int)v78];
      }
      else
      {
        v23 = v79;
        v24 = 0;
        v25 = v79;
      }
      if ( a10 != v22 )
      {
        v26 = v25;
        v63 = v16;
        v27 = a10;
        v28 = v19;
        v61 = v20;
        v29 = v22;
        do
        {
          if ( v26 )
          {
            v30 = *v27;
            *v26 = 6;
            v26[1] = 0;
            v26[2] = v30;
            if ( v30 != 0 && v30 != -8 && v30 != -16 )
              sub_164C220((__int64)v26);
          }
          ++v27;
          v26 += 3;
        }
        while ( v29 != v27 );
        v15 = a10;
        v16 = v63;
        v19 = v28;
        v20 = v61;
        v24 = v78;
        v23 = v77;
      }
      v67 = a11;
      LODWORD(v78) = v24 + v65;
      v31 = (unsigned int)(v24 + v65);
      if ( (unsigned int)a11 >= v20 )
      {
        v59 = v19;
        v32 = v20;
        v33 = 0;
        v62 = v20;
        v66 = v20;
        v64 = v20;
        do
        {
          v34 = &v15[v33];
          v35 = &v23[24 * v33];
          v36 = v34;
          do
          {
            if ( *v36 != *((_QWORD *)v35 + 2) )
              goto LABEL_20;
            ++v36;
            v35 += 24;
          }
          while ( &v34[v66] != v36 );
          v41 = (__int64)&v15[v33];
          sub_1BD8550((__int64)v16, v41, v62, 0, 0, a1);
          if ( (unsigned __int8)sub_1BBD300(v16) )
          {
            v23 = v77;
LABEL_20:
            ++v33;
            goto LABEL_21;
          }
          ++v33;
          sub_1BC4C80(v16, v41, v42, v43, v44, v45);
          v50 = sub_1BD8A90((__int64)v16, v41, v46, v47, v48, v49);
          if ( -dword_4FB9620 > v50 )
          {
            v60 = v50;
            v58 = (_QWORD *)v16[173];
            sub_15CA3B0((__int64)v80, (__int64)"slp-vectorizer", (__int64)"StoresVectorized", 16, *v34);
            sub_15CAB20((__int64)v80, "Stores SLP vectorized with cost ", 0x20u);
            sub_15C9890((__int64)v69, "Cost", 4, v60);
            v51 = sub_17C2270((__int64)v80, (__int64)v69);
            sub_15CAB20(v51, " and with tree size ", 0x14u);
            sub_15C9C50((__int64)v73, "TreeSize", 8, -1171354717 * ((v16[1] - *v16) >> 4));
            v52 = sub_17C2270(v51, (__int64)v73);
            sub_143AA50(v58, v52);
            if ( v75 != &v76 )
              j_j___libc_free_0(v75, v76 + 1);
            if ( (__int64 *)v73[0] != &v74 )
              j_j___libc_free_0(v73[0], v74 + 1);
            if ( v71 != &v72 )
              j_j___libc_free_0(v71, v72 + 1);
            if ( (__int64 *)v69[0] != &v70 )
              j_j___libc_free_0(v69[0], v70 + 1);
            v55 = v81;
            v80[0] = &unk_49ECF68;
            v56 = &v81[88 * v82];
            if ( v81 != v56 )
            {
              do
              {
                v56 -= 88;
                v57 = (char *)*((_QWORD *)v56 + 4);
                if ( v57 != v56 + 48 )
                  j_j___libc_free_0(v57, *((_QWORD *)v56 + 6) + 1LL);
                if ( *(char **)v56 != v56 + 16 )
                  j_j___libc_free_0(*(_QWORD *)v56, *((_QWORD *)v56 + 2) + 1LL);
              }
              while ( v55 != v56 );
              v56 = v81;
            }
            if ( v56 != &v83 )
              _libc_free((unsigned __int64)v56);
            v33 = v32;
            sub_1BD47F0((__int64 ***)v16, (__m128)a1, a2, a3, a4, v53, v54, a7, a8);
            v59 = 1;
          }
          v23 = v77;
LABEL_21:
          v32 = v64 + v33;
        }
        while ( v64 + v33 <= v67 );
        v19 = v59;
        v31 = (unsigned int)v78;
      }
      v37 = &v23[24 * v31];
      if ( v37 != (_QWORD *)v23 )
      {
        v38 = v23;
        v39 = v19;
        do
        {
          v40 = *(v37 - 1);
          v37 -= 3;
          if ( v40 != -8 && v40 != 0 && v40 != -16 )
            sub_1649B30(v37);
        }
        while ( v37 != v38 );
        v23 = v77;
        v19 = v39;
      }
      if ( v23 != v79 )
      {
        v68 = v19;
        _libc_free((unsigned __int64)v23);
        return v68;
      }
    }
  }
  return v19;
}
