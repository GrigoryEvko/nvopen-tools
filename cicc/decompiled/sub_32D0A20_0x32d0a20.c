// Function: sub_32D0A20
// Address: 0x32d0a20
//
__int64 __fastcall sub_32D0A20(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __m128i v7; // xmm0
  __int64 v8; // rcx
  unsigned __int16 *v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rdi
  __m128i v14; // xmm1
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r9d
  __int64 v22; // rdx
  char v23; // al
  __int64 v24; // rcx
  __int64 v25; // rdx
  int v26; // eax
  int v27; // eax
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rax
  unsigned __int16 v31; // dx
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int8 v36; // al
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // r12
  __int64 v41; // rsi
  __int128 v42; // rax
  int v43; // r9d
  __int16 *v44; // rdx
  __int16 v45; // cx
  __int64 v46; // rax
  __int16 v47; // dx
  __int64 v48; // rax
  bool v49; // al
  __int64 v50; // rax
  int v51; // r8d
  unsigned __int64 v52; // rax
  __int64 v53; // rdx
  char v54; // r12
  __int128 v55; // [rsp-20h] [rbp-E0h]
  __int128 v56; // [rsp-10h] [rbp-D0h]
  __int128 v57; // [rsp-10h] [rbp-D0h]
  __int128 v58; // [rsp-10h] [rbp-D0h]
  __int64 v59; // [rsp+8h] [rbp-B8h]
  unsigned int v60; // [rsp+14h] [rbp-ACh]
  __int64 v61; // [rsp+18h] [rbp-A8h]
  __int64 v62; // [rsp+28h] [rbp-98h]
  unsigned int v63; // [rsp+28h] [rbp-98h]
  __m128i v64; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int16 v65[4]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v66; // [rsp+48h] [rbp-78h]
  __int64 v67; // [rsp+50h] [rbp-70h] BYREF
  int v68; // [rsp+58h] [rbp-68h]
  unsigned __int16 v69; // [rsp+60h] [rbp-60h] BYREF
  __int64 v70; // [rsp+68h] [rbp-58h]
  unsigned __int64 v71; // [rsp+70h] [rbp-50h] BYREF
  __int64 v72; // [rsp+78h] [rbp-48h]
  __m128i v73; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_QWORD *)(v4 + 8);
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v62 = *(_QWORD *)v4;
  v8 = *(_QWORD *)(v4 + 40);
  LODWORD(v4) = *(_DWORD *)(v4 + 48);
  v64 = v7;
  v61 = v8;
  v60 = v4;
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v67 = v10;
  v65[0] = v11;
  v66 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v67, v10, 1);
  v13 = *(_QWORD *)a1;
  v68 = *(_DWORD *)(a2 + 72);
  v14 = _mm_load_si128(&v64);
  v71 = v5;
  v72 = v6;
  v73 = v14;
  v15 = sub_3402EA0(v13, 152, (unsigned int)&v67, *(_DWORD *)v65, v66, 0, (__int64)&v71, 2);
  if ( v15 )
  {
    v16 = v15;
    goto LABEL_5;
  }
  v18 = *(_QWORD *)(a2 + 40);
  v19 = *(_QWORD *)(v18 + 48);
  v20 = sub_33E1790(*(_QWORD *)(v18 + 40), v19, 0);
  if ( v20 )
  {
    v59 = *(_QWORD *)(v20 + 96);
    if ( *(void **)(v59 + 24) == sub_C33340() )
      v22 = *(_QWORD *)(v59 + 32);
    else
      v22 = v59 + 24;
    v23 = *(_BYTE *)(a1 + 33);
    if ( (*(_BYTE *)(v22 + 20) & 8) != 0 )
    {
      if ( !v23
        || ((v38 = *(_QWORD *)(a1 + 8), v39 = 1, v65[0] == 1)
         || v65[0] && (v39 = v65[0], *(_QWORD *)(v38 + 8LL * v65[0] + 112)))
        && !*(_BYTE *)(v38 + 500 * v39 + 6658) )
      {
        v40 = *(_QWORD *)a1;
        v41 = *(_QWORD *)(v62 + 80);
        v71 = v41;
        if ( v41 )
          sub_B96E90((__int64)&v71, v41, 1);
        *((_QWORD *)&v57 + 1) = v6;
        *(_QWORD *)&v57 = v5;
        LODWORD(v72) = *(_DWORD *)(v62 + 72);
        *(_QWORD *)&v42 = sub_33FAF80(v40, 245, (unsigned int)&v71, *(_DWORD *)v65, v66, v21, v57);
        v16 = sub_33FAF80(v40, 244, (unsigned int)&v67, *(_DWORD *)v65, v66, v43, v42);
        if ( v71 )
          sub_B91220((__int64)&v71, v71);
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v23 )
        goto LABEL_17;
      v24 = *(_QWORD *)(a1 + 8);
      v25 = 1;
      if ( v65[0] == 1 || v65[0] && (v25 = v65[0], *(_QWORD *)(v24 + 8LL * v65[0] + 112)) )
      {
        if ( !*(_BYTE *)(v24 + 500 * v25 + 6659) )
          goto LABEL_17;
      }
    }
  }
  v26 = *(_DWORD *)(v62 + 24);
  if ( (unsigned int)(v26 - 244) <= 1 || v26 == 152 )
  {
    v37 = sub_3406EB0(
            *(_QWORD *)a1,
            152,
            (unsigned int)&v67,
            *(_DWORD *)v65,
            v66,
            v21,
            *(_OWORD *)*(_QWORD *)(v62 + 40),
            *(_OWORD *)&v64);
LABEL_33:
    v16 = v37;
    goto LABEL_5;
  }
  v27 = *(_DWORD *)(v61 + 24);
  if ( v27 == 245 )
  {
LABEL_17:
    *((_QWORD *)&v56 + 1) = v6;
    *(_QWORD *)&v56 = v5;
    v16 = sub_33FAF80(*(_QWORD *)a1, 245, (unsigned int)&v67, *(_DWORD *)v65, v66, v21, v56);
    goto LABEL_5;
  }
  if ( v27 == 152 )
  {
    v58 = *(_OWORD *)(*(_QWORD *)(v61 + 40) + 40LL);
    goto LABEL_47;
  }
  v28 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  v29 = *(_DWORD *)(v28 + 24);
  if ( v29 == 230 || v29 == 233 )
  {
    v44 = *(__int16 **)(v28 + 48);
    v45 = *v44;
    v19 = *((_QWORD *)v44 + 1);
    v46 = *(_QWORD *)(**(_QWORD **)(v28 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v28 + 40) + 8LL);
    v47 = *(_WORD *)v46;
    v48 = *(_QWORD *)(v46 + 8);
    LOWORD(v71) = v47;
    v72 = v48;
    if ( v45 == v47 )
    {
      if ( v45 || v19 == v48 )
      {
LABEL_46:
        v58 = *(_OWORD *)*(_QWORD *)(v61 + 40);
LABEL_47:
        *((_QWORD *)&v55 + 1) = v6;
        *(_QWORD *)&v55 = v5;
        v37 = sub_3406EB0(*(_QWORD *)a1, 152, (unsigned int)&v67, *(_DWORD *)v65, v66, v21, v55, v58);
        goto LABEL_33;
      }
    }
    else
    {
      if ( v47 == 15 )
        goto LABEL_24;
      if ( v47 )
      {
        v49 = (unsigned __int16)(v47 - 17) <= 0xD3u;
        goto LABEL_52;
      }
    }
    v49 = sub_30070B0((__int64)&v71);
LABEL_52:
    if ( v49 )
      goto LABEL_24;
    goto LABEL_46;
  }
LABEL_24:
  v30 = *(_QWORD *)(v61 + 48) + 16LL * v60;
  v31 = *(_WORD *)v30;
  v32 = *(_QWORD *)(v30 + 8);
  v69 = v31;
  v70 = v32;
  v33 = sub_32844A0(&v69, v19);
  LODWORD(v72) = v33;
  if ( v33 > 0x40 )
  {
    v63 = v33;
    sub_C43690((__int64)&v71, 0, 0);
    v34 = 1LL << ((unsigned __int8)v63 - 1);
    if ( (unsigned int)v72 > 0x40 )
    {
      *(_QWORD *)(v71 + 8LL * ((v63 - 1) >> 6)) |= v34;
      goto LABEL_27;
    }
  }
  else
  {
    v71 = 0;
    v34 = 1LL << ((unsigned __int8)v33 - 1);
  }
  v71 |= v34;
LABEL_27:
  v35 = v64.m128i_i64[0];
  v36 = sub_32D08B0(a1, v64.m128i_i64[0], v64.m128i_u32[2], (int)&v71);
  if ( (unsigned int)v72 > 0x40 && v71 )
  {
    v64.m128i_i8[0] = v36;
    j_j___libc_free_0_0(v71);
    v36 = v64.m128i_i8[0];
  }
  if ( v36 )
    goto LABEL_31;
  v50 = sub_32844A0(v65, v35);
  LODWORD(v72) = v50;
  v51 = v50;
  if ( (unsigned int)v50 <= 0x40 )
  {
    v52 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v50;
    if ( !v51 )
      v52 = 0;
    v71 = v52;
    v53 = ~(1LL << ((unsigned __int8)v51 - 1));
    goto LABEL_60;
  }
  v64.m128i_i64[0] = v50;
  sub_C43690((__int64)&v71, -1, 1);
  v53 = ~(1LL << (v64.m128i_i8[0] - 1));
  if ( (unsigned int)v72 <= 0x40 )
  {
LABEL_60:
    v71 &= v53;
    goto LABEL_61;
  }
  *(_QWORD *)(v71 + 8LL * ((unsigned int)(v64.m128i_i32[0] - 1) >> 6)) &= v53;
LABEL_61:
  v54 = sub_32D08B0(a1, v5, v6, (int)&v71);
  if ( (unsigned int)v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  if ( v54 )
  {
LABEL_31:
    v16 = a2;
    goto LABEL_5;
  }
  v16 = 0;
LABEL_5:
  if ( v67 )
    sub_B91220((__int64)&v67, v67);
  return v16;
}
