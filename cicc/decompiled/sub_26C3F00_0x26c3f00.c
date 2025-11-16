// Function: sub_26C3F00
// Address: 0x26c3f00
//
__int64 __fastcall sub_26C3F00(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // eax
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 *v17; // r13
  unsigned int v18; // r15d
  __int64 v19; // rax
  _BYTE *v20; // rdi
  _QWORD *v21; // rbx
  _QWORD *v22; // r12
  __int64 v23; // rax
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // r14
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 *v31; // r13
  __int64 v32; // rax
  __int64 *v33; // rax
  _QWORD *v34; // r15
  __int64 *v35; // rbx
  __int64 v36; // r14
  _QWORD *v37; // rbx
  _QWORD *v38; // r13
  __int64 v39; // [rsp-8h] [rbp-2C8h]
  int v40; // [rsp+8h] [rbp-2B8h]
  __int64 v41; // [rsp+8h] [rbp-2B8h]
  __int64 v42; // [rsp+30h] [rbp-290h]
  __int64 v43; // [rsp+38h] [rbp-288h]
  int v44; // [rsp+40h] [rbp-280h]
  __int64 v46; // [rsp+58h] [rbp-268h] BYREF
  _BYTE v47[12]; // [rsp+60h] [rbp-260h] BYREF
  char *v48; // [rsp+70h] [rbp-250h]
  unsigned __int64 v49; // [rsp+78h] [rbp-248h] BYREF
  unsigned int v50; // [rsp+80h] [rbp-240h]
  unsigned __int64 v51; // [rsp+88h] [rbp-238h] BYREF
  unsigned int v52; // [rsp+90h] [rbp-230h]
  char v53; // [rsp+98h] [rbp-228h]
  __m128i v54; // [rsp+A0h] [rbp-220h] BYREF
  char *v55; // [rsp+B0h] [rbp-210h]
  unsigned __int64 v56; // [rsp+B8h] [rbp-208h]
  unsigned int v57; // [rsp+C0h] [rbp-200h]
  unsigned __int64 v58; // [rsp+C8h] [rbp-1F8h]
  unsigned int v59; // [rsp+D0h] [rbp-1F0h]
  char v60; // [rsp+D8h] [rbp-1E8h]
  __int64 (__fastcall *v61)(__int64, __int64, __int64); // [rsp+E0h] [rbp-1E0h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-1D8h]
  char *v63; // [rsp+F0h] [rbp-1D0h]
  unsigned __int64 v64; // [rsp+F8h] [rbp-1C8h] BYREF
  __int64 v65; // [rsp+100h] [rbp-1C0h]
  _BYTE *v66; // [rsp+108h] [rbp-1B8h] BYREF
  __int64 v67; // [rsp+110h] [rbp-1B0h]
  _BYTE v68[4]; // [rsp+118h] [rbp-1A8h] BYREF
  __int16 v69; // [rsp+11Ch] [rbp-1A4h]
  char v70; // [rsp+120h] [rbp-1A0h]
  char v71; // [rsp+121h] [rbp-19Fh]
  char v72[8]; // [rsp+130h] [rbp-190h] BYREF
  _BYTE *v73; // [rsp+138h] [rbp-188h]
  __int64 v74; // [rsp+140h] [rbp-180h]
  _BYTE v75[192]; // [rsp+148h] [rbp-178h] BYREF
  _BYTE *v76; // [rsp+208h] [rbp-B8h]
  __int64 v77; // [rsp+210h] [rbp-B0h]
  _BYTE v78[168]; // [rsp+218h] [rbp-A8h] BYREF

  v4 = (_QWORD *)a1;
  v5 = *(_QWORD **)a2;
  v6 = *(_QWORD *)(*(_QWORD *)a2 - 32LL);
  v44 = v6;
  if ( v6 )
  {
    if ( *(_BYTE *)v6 )
    {
      v44 = 0;
    }
    else
    {
      LODWORD(v7) = 0;
      if ( *(_QWORD *)(v6 + 24) == v5[10] )
        v7 = *(_QWORD *)(*(_QWORD *)a2 - 32LL);
      v44 = v7;
    }
  }
  v8 = v5[6];
  v9 = (__int64)v5;
  v46 = v8;
  if ( v8 )
  {
    sub_B96E90((__int64)&v46, v8, 1);
    v9 = *(_QWORD *)a2;
  }
  v43 = v5[5];
  sub_26C3D10((__int64)&v61, a1, v9);
  if ( v70 )
  {
    v53 = 0;
    *(_QWORD *)v47 = v61;
    *(_DWORD *)&v47[8] = v62;
    v48 = v63;
    if ( v68[0] )
    {
      v50 = v65;
      if ( (unsigned int)v65 > 0x40 )
        sub_C43780((__int64)&v49, (const void **)&v64);
      else
        v49 = v64;
      v52 = v67;
      if ( (unsigned int)v67 > 0x40 )
        sub_C43780((__int64)&v51, (const void **)&v66);
      else
        v51 = (unsigned __int64)v66;
      v53 = 1;
      if ( v70 )
      {
        v70 = 0;
        if ( v68[0] )
          sub_26C3CC0((__int64)&v64);
      }
    }
    goto LABEL_61;
  }
  v40 = qword_4FF7320[17];
  if ( (_BYTE)qword_4FF6E68 )
  {
    v15 = *(_QWORD *)(a1 + 1280);
    v10 = *(_QWORD *)(a2 + 16);
    v16 = 0;
    if ( *(_BYTE *)(v15 + 24) )
      v16 = *(_QWORD *)(v15 + 16);
    if ( v10 > v16 )
    {
      v40 = qword_4FF7400[17];
    }
    else if ( !byte_4FF79C8 )
    {
      v53 = 0;
      *(_QWORD *)v47 = 0x7FFFFFFF;
      *(_DWORD *)&v47[8] = 0;
      v48 = "cold callsite";
      goto LABEL_33;
    }
  }
  v11 = *(_QWORD *)(*(_QWORD *)a2 - 32LL);
  if ( v11 )
  {
    if ( *(_BYTE *)v11 )
    {
      v11 = 0;
    }
    else if ( *(_QWORD *)(v11 + 24) != *(_QWORD *)(*(_QWORD *)a2 + 80LL) )
    {
      v11 = 0;
    }
  }
  v42 = v11;
  sub_30D6B30(&v61, a1, v11, v10);
  v69 = 257;
  v70 = qword_4FF6CA8;
  if ( !v71 )
    v71 = 1;
  if ( !*(_QWORD *)(a1 + 1456) )
    sub_4263D6(&v61, a1, v42);
  v12 = (*(__int64 (__fastcall **)(__int64, __int64))(a1 + 1464))(a1 + 1440, v42);
  sub_30DEDC0(
    (unsigned int)&v54,
    *(_QWORD *)a2,
    v42,
    (unsigned int)&v61,
    v12,
    0,
    (__int64)sub_26B9F70,
    a1 + 1408,
    (__int64)sub_24258E0,
    a1 + 1472,
    0,
    0,
    0);
  v13 = v54.m128i_i32[0];
  if ( (unsigned int)(v54.m128i_i32[0] + 0x7FFFFFFF) > 0xFFFFFFFD )
  {
    *(_DWORD *)v47 = v54.m128i_i32[0];
    v53 = 0;
    *(_QWORD *)&v47[4] = *(__int64 *)((char *)v54.m128i_i64 + 4);
    v48 = v55;
    if ( !v60 )
      goto LABEL_41;
    v26 = v57;
    v53 = 1;
    v57 = 0;
    v50 = v26;
    v60 = 0;
    v49 = v56;
    v52 = v59;
    v51 = v58;
    goto LABEL_67;
  }
  if ( (_BYTE)qword_4FF6D88
    && (v14 = *(_QWORD *)(a2 + 8)) != 0
    && (*(_BYTE *)(v14 + 48) & 2) == 0
    && (*(_BYTE *)(v14 + 52) & 2) != 0 )
  {
    v53 = 0;
    *(_QWORD *)v47 = 0x80000000LL;
    *(_DWORD *)&v47[8] = 0;
    v48 = "preinliner";
  }
  else
  {
    *(_DWORD *)v47 = v54.m128i_i32[0];
    if ( (_BYTE)qword_4FF6E68 )
    {
      *(_DWORD *)&v47[8] = 0;
      *(_DWORD *)&v47[4] = v40;
      v48 = 0;
      v53 = 0;
    }
    else
    {
      v25 = qword_4FF7400[17];
      v53 = 0;
      *(_DWORD *)&v47[8] = 0;
      *(_DWORD *)&v47[4] = v25;
      v48 = 0;
    }
  }
  if ( v60 )
  {
    v60 = 0;
    if ( v59 > 0x40 && v58 )
      j_j___libc_free_0_0(v58);
LABEL_67:
    if ( v57 > 0x40 && v56 )
      j_j___libc_free_0_0(v56);
  }
LABEL_61:
  v13 = *(_DWORD *)v47;
LABEL_41:
  if ( v13 == 0x7FFFFFFF )
  {
LABEL_33:
    v17 = *(__int64 **)(a1 + 1288);
    v18 = 0;
    sub_B157E0((__int64)&v54, &v46);
    sub_B17850((__int64)&v61, *(_QWORD *)(a1 + 1528), (__int64)"InlineFail", 10, &v54, v43);
    sub_B18290((__int64)&v61, "incompatible inlining", 0x15u);
    sub_1049740(v17, (__int64)&v61);
    v61 = (__int64 (__fastcall *)(__int64, __int64, __int64))&unk_49D9D40;
    sub_23FD590((__int64)v72);
    if ( !v53 )
      goto LABEL_57;
    goto LABEL_34;
  }
  v18 = 0;
  if ( *(int *)&v47[4] > v13 )
  {
    v63 = 0;
    v61 = sub_26B9F70;
    v62 = a1 + 1408;
    v66 = v68;
    v67 = 0x400000000LL;
    v73 = v75;
    v76 = v78;
    v64 = 0;
    v65 = 0;
    v74 = 0x800000000LL;
    v77 = 0x800000000LL;
    v78[64] = 0;
    v19 = sub_29F2700(v5, &v61, 1, 0, 1, 0);
    v20 = v76;
    if ( !v19 )
    {
      v27 = v4[191];
      v28 = *(_QWORD *)(v43 + 72);
      v54.m128i_i64[0] = v46;
      if ( v46 )
      {
        v41 = v27;
        sub_B96E90((__int64)&v54, v46, 1);
        v27 = v41;
      }
      sub_30CD680(v4[161], (unsigned int)&v54, v43, v44, v28, (unsigned int)v47, 1, v27);
      v30 = v39;
      if ( v54.m128i_i64[0] )
        sub_B91220((__int64)&v54, v54.m128i_i64[0]);
      if ( a3 )
      {
        v31 = (__int64 *)v76;
        v32 = (unsigned int)v77;
        *(_DWORD *)(a3 + 8) = 0;
        v33 = &v31[v32];
        if ( v31 != v33 )
        {
          v30 = 0;
          v34 = v4;
          v35 = v33;
          do
          {
            v36 = *v31;
            if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            {
              sub_C8D5F0(a3, (const void *)(a3 + 16), v30 + 1, 8u, v30 + 1, v29);
              v30 = *(unsigned int *)(a3 + 8);
            }
            ++v31;
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v30) = v36;
            v30 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
            *(_DWORD *)(a3 + 8) = v30;
          }
          while ( v35 != v31 );
          v4 = v34;
        }
      }
      if ( unk_4F838D3 )
        sub_317E680(v4[189], *(_QWORD *)(a2 + 8), v30);
      v20 = v76;
      if ( *(float *)(a2 + 24) >= 1.0 || (v37 = &v76[8 * (unsigned int)v77], v37 == (_QWORD *)v76) )
      {
        v18 = 1;
      }
      else
      {
        v38 = v76;
        do
        {
          sub_3143F80(&v54, *v38, v30);
          if ( BYTE4(v55) )
            sub_3144140(*v38, *(float *)&v55 * *(float *)(a2 + 24));
          ++v38;
        }
        while ( v37 != v38 );
        v20 = v76;
        v18 = 1;
      }
    }
    if ( v20 != v78 )
      _libc_free((unsigned __int64)v20);
    v21 = v73;
    v22 = &v73[24 * (unsigned int)v74];
    if ( v73 != (_BYTE *)v22 )
    {
      do
      {
        v23 = *(v22 - 1);
        v22 -= 3;
        if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
          sub_BD60C0(v22);
      }
      while ( v21 != v22 );
      v22 = v73;
    }
    if ( v22 != (_QWORD *)v75 )
      _libc_free((unsigned __int64)v22);
    if ( v66 != v68 )
      _libc_free((unsigned __int64)v66);
  }
  if ( v53 )
  {
LABEL_34:
    v53 = 0;
    if ( v52 > 0x40 && v51 )
      j_j___libc_free_0_0(v51);
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
  }
LABEL_57:
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v18;
}
