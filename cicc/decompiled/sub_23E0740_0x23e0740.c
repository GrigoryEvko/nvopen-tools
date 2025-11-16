// Function: sub_23E0740
// Address: 0x23e0740
//
void __fastcall sub_23E0740(__int64 *a1)
{
  unsigned int v1; // r15d
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int16 v10; // ax
  char v11; // al
  __int64 v12; // rdx
  __m128i v13; // xmm1
  __int64 v14; // r12
  unsigned __int8 v15; // al
  unsigned int v16; // r12d
  unsigned __int8 v17; // r13
  _QWORD *v18; // r14
  unsigned __int64 v19; // r13
  _BYTE *v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  char v23; // r12
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // edx
  __int64 v30; // r8
  unsigned int v31; // ecx
  __int32 v32; // eax
  __m128i *v33; // rdx
  char v34; // cl
  _QWORD *v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // [rsp-1D0h] [rbp-1D0h]
  __int64 v40; // [rsp-1C8h] [rbp-1C8h]
  unsigned __int8 v41; // [rsp-1ABh] [rbp-1ABh]
  unsigned __int16 v42; // [rsp-1AAh] [rbp-1AAh]
  __int64 v43; // [rsp-1A0h] [rbp-1A0h]
  __int64 v44; // [rsp-198h] [rbp-198h]
  __int64 *v45; // [rsp-190h] [rbp-190h]
  __m128i v46; // [rsp-188h] [rbp-188h] BYREF
  __m128i v47; // [rsp-178h] [rbp-178h] BYREF
  __int64 v48; // [rsp-168h] [rbp-168h]
  _QWORD v49[4]; // [rsp-158h] [rbp-158h] BYREF
  char v50; // [rsp-138h] [rbp-138h]
  char v51; // [rsp-137h] [rbp-137h]
  __m128i v52; // [rsp-128h] [rbp-128h] BYREF
  __m128i v53; // [rsp-118h] [rbp-118h]
  __int64 v54; // [rsp-108h] [rbp-108h]
  _QWORD v55[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v56; // [rsp-D8h] [rbp-D8h]
  _BYTE *v57; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v58; // [rsp-C0h] [rbp-C0h]
  _BYTE v59[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v60; // [rsp-98h] [rbp-98h]
  __int64 v61; // [rsp-90h] [rbp-90h]
  __int64 v62; // [rsp-88h] [rbp-88h]
  _QWORD *v63; // [rsp-80h] [rbp-80h]
  void **v64; // [rsp-78h] [rbp-78h]
  void **v65; // [rsp-70h] [rbp-70h]
  __int64 v66; // [rsp-68h] [rbp-68h]
  int v67; // [rsp-60h] [rbp-60h]
  __int16 v68; // [rsp-5Ch] [rbp-5Ch]
  char v69; // [rsp-5Ah] [rbp-5Ah]
  __int64 v70; // [rsp-58h] [rbp-58h]
  __int64 v71; // [rsp-50h] [rbp-50h]
  void *v72; // [rsp-48h] [rbp-48h] BYREF
  void *v73; // [rsp-40h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(*a1 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  if ( v3 )
    v3 -= 24;
  if ( *(_QWORD *)(a1[1] + 1016) == v3 )
  {
    v38 = *(_QWORD *)(v3 + 32);
    if ( v38 == *(_QWORD *)(v3 + 40) + 48LL || !v38 )
      v3 = 0;
    else
      v3 = v38 - 24;
  }
  v69 = 7;
  v63 = (_QWORD *)sub_BD5C60(v3);
  v64 = &v72;
  v65 = &v73;
  v58 = 0x200000000LL;
  v4 = v3;
  v72 = &unk_49DA100;
  v57 = v59;
  v68 = 512;
  LOWORD(v62) = 0;
  v73 = &unk_49DA0B0;
  v66 = 0;
  v67 = 0;
  v70 = 0;
  v71 = 0;
  v60 = 0;
  v61 = 0;
  sub_D5F1F0((__int64)&v57, v3);
  v5 = sub_B2BEC0(*a1);
  v8 = *a1;
  v43 = v5;
  if ( (*(_BYTE *)(*a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v8, v4, v6, v7);
    v9 = *(_QWORD *)(v8 + 96);
    v44 = v9 + 40LL * *(_QWORD *)(v8 + 104);
    if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v8, v4, v36, v37);
      v9 = *(_QWORD *)(v8 + 96);
    }
  }
  else
  {
    v9 = *(_QWORD *)(v8 + 96);
    v44 = v9 + 40LL * *(_QWORD *)(v8 + 104);
  }
  if ( v44 != v9 )
  {
    while ( 1 )
    {
      while ( !(unsigned __int8)sub_B2D680(v9) )
      {
        v9 += 40;
        if ( v44 == v9 )
          goto LABEL_23;
      }
      v45 = (__int64 *)sub_B2BD20(v9);
      v10 = sub_B2BD00(v9);
      if ( !HIBYTE(v10) )
        LOBYTE(v10) = sub_AE5020(v43, (__int64)v45);
      v41 = v10;
      v51 = 1;
      v49[0] = ".byval";
      v50 = 3;
      if ( (*(_BYTE *)(v9 + 7) & 0x10) == 0 )
        break;
      v46.m128i_i64[0] = (__int64)sub_BD5D20(v9);
      v11 = v50;
      LOWORD(v48) = 261;
      v46.m128i_i64[1] = v12;
      if ( v50 )
      {
        if ( v50 != 1 )
        {
          v33 = (__m128i *)v46.m128i_i64[0];
          v34 = 5;
          v39 = v46.m128i_i64[1];
          goto LABEL_27;
        }
        v13 = _mm_loadu_si128(&v47);
        v52 = _mm_loadu_si128(&v46);
        v54 = v48;
        v53 = v13;
      }
      else
      {
        LOWORD(v54) = 256;
      }
LABEL_18:
      v14 = sub_AA4E30(v60);
      v15 = sub_AE5260(v14, (__int64)v45);
      v16 = *(_DWORD *)(v14 + 4);
      v17 = v15;
      v56 = 257;
      v18 = sub_BD2C40(80, unk_3F10A14);
      if ( v18 )
        sub_B4CCA0((__int64)v18, v45, v16, 0, v17, (__int64)v55, 0, 0);
      (*((void (__fastcall **)(void **, _QWORD *, __m128i *, __int64, __int64))*v65 + 2))(v65, v18, &v52, v61, v62);
      v19 = (unsigned __int64)v57;
      v20 = &v57[16 * (unsigned int)v58];
      if ( v57 != v20 )
      {
        do
        {
          v21 = *(_QWORD *)(v19 + 8);
          v22 = *(_DWORD *)v19;
          v19 += 16LL;
          sub_B99FD0((__int64)v18, v22, v21);
        }
        while ( v20 != (_BYTE *)v19 );
      }
      LOBYTE(v1) = v41;
      *((_WORD *)v18 + 1) = v41 | *((_WORD *)v18 + 1) & 0xFFC0;
      sub_BD84D0(v9, (__int64)v18);
      v23 = sub_AE5020(v43, (__int64)v45);
      v24 = sub_9208B0(v43, (__int64)v45);
      v55[1] = v25;
      v55[0] = ((1LL << v23) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v23 << v23;
      v26 = sub_CA1930(v55);
      v27 = sub_BCB2E0(v63);
      v28 = sub_ACD640(v27, v26, 0);
      v29 = v1;
      v30 = v9;
      v31 = v42;
      BYTE1(v29) = 1;
      v1 = v29;
      LOBYTE(v31) = v41;
      v9 += 40;
      BYTE1(v31) = 1;
      v42 = v31;
      sub_B343C0((__int64)&v57, 0xEEu, (__int64)v18, v31, v30, v29, v28, 0, 0, 0, 0, 0);
      if ( v44 == v9 )
        goto LABEL_23;
    }
    v32 = *(_DWORD *)(v9 + 32);
    v33 = &v46;
    v46.m128i_i64[0] = (__int64)"Arg";
    v34 = 2;
    LOWORD(v48) = 2307;
    v47.m128i_i32[0] = v32;
    v11 = 3;
LABEL_27:
    if ( v51 == 1 )
    {
      v40 = v49[1];
      v35 = (_QWORD *)v49[0];
    }
    else
    {
      v35 = v49;
      v11 = 2;
    }
    v53.m128i_i64[0] = (__int64)v35;
    v52.m128i_i64[0] = (__int64)v33;
    v52.m128i_i64[1] = v39;
    v53.m128i_i64[1] = v40;
    LOBYTE(v54) = v34;
    BYTE1(v54) = v11;
    goto LABEL_18;
  }
LABEL_23:
  nullsub_61();
  v72 = &unk_49DA100;
  nullsub_63();
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
}
