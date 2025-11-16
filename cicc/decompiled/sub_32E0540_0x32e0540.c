// Function: sub_32E0540
// Address: 0x32e0540
//
__int64 __fastcall sub_32E0540(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // r15
  unsigned __int16 *v8; // rax
  __int64 v9; // rcx
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  bool v16; // zf
  unsigned __int16 v17; // r8
  __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // rsi
  unsigned int v21; // edx
  unsigned int v22; // edx
  bool v23; // al
  int v24; // r9d
  int v25; // ecx
  bool v26; // r12
  __int64 v27; // rax
  bool v28; // al
  int v29; // eax
  __int64 v30; // rax
  int v31; // edx
  __int64 *v32; // r12
  __int64 v33; // rdi
  __int64 v34; // r12
  __int64 v35; // r12
  __int128 v36; // rax
  int v37; // r9d
  __int64 *v38; // r12
  __int64 v39; // rdi
  __int128 v40; // rax
  int v41; // r9d
  int v42; // [rsp+18h] [rbp-128h]
  unsigned __int16 v43; // [rsp+20h] [rbp-120h]
  bool v44; // [rsp+28h] [rbp-118h]
  int v45; // [rsp+30h] [rbp-110h]
  __int64 v46; // [rsp+40h] [rbp-100h]
  int v47; // [rsp+40h] [rbp-100h]
  __int64 v48; // [rsp+48h] [rbp-F8h]
  __m128i v49; // [rsp+50h] [rbp-F0h]
  __m128i v50; // [rsp+60h] [rbp-E0h]
  __int64 *v51; // [rsp+60h] [rbp-E0h]
  __int64 *v52; // [rsp+60h] [rbp-E0h]
  __int128 v53; // [rsp+60h] [rbp-E0h]
  __int64 v54; // [rsp+70h] [rbp-D0h] BYREF
  int v55; // [rsp+78h] [rbp-C8h]
  __int128 v56; // [rsp+80h] [rbp-C0h] BYREF
  __int128 v57; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+A0h] [rbp-A0h] BYREF
  int v59; // [rsp+A8h] [rbp-98h]
  __int128 *v60; // [rsp+B0h] [rbp-90h]
  __int128 *v61; // [rsp+B8h] [rbp-88h]
  char v62; // [rsp+C4h] [rbp-7Ch]
  int v63; // [rsp+C8h] [rbp-78h]
  int v64; // [rsp+D0h] [rbp-70h]
  __int128 *v65; // [rsp+D8h] [rbp-68h]
  __int128 *v66; // [rsp+E0h] [rbp-60h]
  char v67; // [rsp+ECh] [rbp-54h]
  unsigned __int64 v68; // [rsp+F0h] [rbp-50h]
  __int64 v69; // [rsp+F8h] [rbp-48h]
  char v70; // [rsp+104h] [rbp-3Ch]
  char v71; // [rsp+10Ch] [rbp-34h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = v4[5];
  v50 = _mm_loadu_si128((const __m128i *)v4);
  v49 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v8 = (unsigned __int16 *)(*(_QWORD *)(*v4 + 48) + 16LL * *((unsigned int *)v4 + 2));
  v9 = *((_QWORD *)v8 + 1);
  v10 = *v8;
  v54 = v5;
  v48 = v9;
  if ( v5 )
    sub_B96E90((__int64)&v54, v5, 1);
  v55 = *(_DWORD *)(a2 + 72);
  v11 = sub_32DEAD0(a1, a2);
  if ( v11 )
    goto LABEL_4;
  v11 = sub_3270B90(*(_DWORD *)(a2 + 24), *(__int64 **)(a2 + 40), (__int64)&v54, *a1);
  if ( v11 )
    goto LABEL_4;
  v11 = sub_327A780(*(_DWORD *)(a2 + 24), *(_QWORD **)(a2 + 40), (int)&v54, *a1);
  if ( v11 )
    goto LABEL_4;
  if ( (_BYTE)qword_5037C28 )
  {
    v14 = *(_QWORD *)(a2 + 80);
    v58 = v14;
    if ( v14 )
      sub_B96E90((__int64)&v58, v14, 1);
    v59 = *(_DWORD *)(a2 + 72);
    v11 = sub_329F5D0(
            (__int64)a1,
            v50.m128i_i64[0],
            v50.m128i_i64[1],
            v49.m128i_i64[0],
            v49.m128i_i64[1],
            (__int64)&v58,
            1);
    if ( v58 )
    {
      v46 = v11;
      sub_B91220((__int64)&v58, v58);
      v11 = v46;
    }
    if ( v11 )
    {
LABEL_4:
      v12 = v11;
      goto LABEL_5;
    }
  }
  v15 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v16 = *((_BYTE *)a1 + 33) == 0;
  v17 = *(_WORD *)v15;
  v18 = *(_QWORD *)(v15 + 8);
  *(_QWORD *)&v56 = 0;
  DWORD2(v56) = 0;
  v47 = v18;
  v19 = v17;
  *(_QWORD *)&v57 = 0;
  DWORD2(v57) = 0;
  if ( !v16 )
  {
    v20 = a1[1];
    v21 = 1;
    if ( v17 != 1 )
    {
      if ( !v17 )
        goto LABEL_37;
      v21 = v17;
      if ( !*(_QWORD *)(v20 + 8LL * v17 + 112) )
        goto LABEL_21;
    }
    if ( *(_BYTE *)(v20 + 500LL * v21 + 6589) )
      goto LABEL_21;
  }
  v62 = 0;
  v42 = v17;
  v43 = v17;
  v60 = &v56;
  v65 = &v56;
  LODWORD(v58) = 56;
  v59 = 186;
  v61 = &v57;
  v63 = 192;
  v64 = 188;
  v66 = &v57;
  v67 = 0;
  v68 = 1;
  v69 = 64;
  v70 = 0;
  v71 = 0;
  v28 = sub_329DEC0(a2, 0, (__int64)&v58);
  v17 = v43;
  v19 = v42;
  if ( (unsigned int)v69 > 0x40 && v68 )
  {
    v44 = v28;
    j_j___libc_free_0_0(v68);
    v19 = v42;
    v28 = v44;
    v17 = v43;
  }
  if ( !v28 )
  {
    if ( !*((_BYTE *)a1 + 33) )
    {
LABEL_25:
      v45 = v19;
      LODWORD(v58) = 56;
      v59 = 186;
      v60 = &v56;
      v61 = &v57;
      v62 = 0;
      v63 = 191;
      v64 = 188;
      v65 = &v56;
      v66 = &v57;
      v67 = 0;
      v68 = 1;
      v69 = 64;
      v70 = 0;
      v71 = 0;
      v23 = sub_329DEC0(a2, 0, (__int64)&v58);
      v25 = v45;
      v26 = v23;
      if ( (unsigned int)v69 > 0x40 && v68 )
      {
        j_j___libc_free_0_0(v68);
        v25 = v45;
      }
      if ( v26 )
      {
        v27 = sub_3406EB0(*a1, 174, (unsigned int)&v54, v25, v47, v24, v56, v57);
        goto LABEL_30;
      }
LABEL_37:
      if ( (!*((_BYTE *)a1 + 33) || sub_328D6E0(a1[1], 0xBBu, v10))
        && (unsigned __int8)sub_33E0180(*a1, v50.m128i_i64[0], v50.m128i_i64[1], v49.m128i_i64[0], v49.m128i_i64[1]) )
      {
        v12 = sub_3405C90(*a1, 187, (unsigned int)&v54, v10, v48, 8, *(_OWORD *)&v50, *(_OWORD *)&v49);
        goto LABEL_5;
      }
      v29 = *(_DWORD *)(v6 + 24);
      if ( v29 == 373 )
      {
        if ( *(_DWORD *)(v7 + 24) == 373 )
        {
          v34 = *a1;
          v51 = (__int64 *)(*(_QWORD *)(**(_QWORD **)(v7 + 40) + 96LL) + 24LL);
          sub_9865C0((__int64)&v57, *(_QWORD *)(**(_QWORD **)(v6 + 40) + 96LL) + 24LL);
          sub_C45EE0((__int64)&v57, v51);
          v59 = DWORD2(v57);
          DWORD2(v57) = 0;
          v58 = v57;
          v12 = sub_3401900(v34, &v54, v10, v48, &v58, 1);
          sub_969240(&v58);
          sub_969240((__int64 *)&v57);
          goto LABEL_5;
        }
      }
      else if ( v29 == 56 )
      {
        v30 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL);
        v31 = *(_DWORD *)(v30 + 24);
        if ( v31 == 373 )
        {
          if ( *(_DWORD *)(v7 + 24) == 373 )
          {
            v35 = *a1;
            v52 = (__int64 *)(*(_QWORD *)(**(_QWORD **)(v7 + 40) + 96LL) + 24LL);
            sub_9865C0((__int64)&v57, *(_QWORD *)(**(_QWORD **)(v30 + 40) + 96LL) + 24LL);
            sub_C45EE0((__int64)&v57, v52);
            v59 = DWORD2(v57);
            v58 = v57;
            DWORD2(v57) = 0;
            *(_QWORD *)&v36 = sub_3401900(v35, &v54, v10, v48, &v58, 1);
            v53 = v36;
            sub_969240(&v58);
            sub_969240((__int64 *)&v57);
            v12 = sub_3406EB0(*a1, 56, (unsigned int)&v54, v10, v48, v37, *(_OWORD *)*(_QWORD *)(v6 + 40), v53);
            goto LABEL_5;
          }
        }
        else if ( v31 == 170 && *(_DWORD *)(v7 + 24) == 170 )
        {
          v38 = (__int64 *)(*(_QWORD *)(**(_QWORD **)(v7 + 40) + 96LL) + 24LL);
          sub_9865C0((__int64)&v58, *(_QWORD *)(**(_QWORD **)(v30 + 40) + 96LL) + 24LL);
          sub_C45EE0((__int64)&v58, v38);
          v39 = *a1;
          DWORD2(v57) = v59;
          *(_QWORD *)&v57 = v58;
          *(_QWORD *)&v40 = sub_3402600(v39, &v54, v10, v48, &v57);
          v12 = sub_3406EB0(*a1, 56, (unsigned int)&v54, v10, v48, v41, *(_OWORD *)*(_QWORD *)(v6 + 40), v40);
          sub_969240((__int64 *)&v57);
          goto LABEL_5;
        }
      }
      else if ( v29 == 170 && *(_DWORD *)(v7 + 24) == 170 )
      {
        v32 = (__int64 *)(*(_QWORD *)(**(_QWORD **)(v7 + 40) + 96LL) + 24LL);
        sub_9865C0((__int64)&v58, *(_QWORD *)(**(_QWORD **)(v6 + 40) + 96LL) + 24LL);
        sub_C45EE0((__int64)&v58, v32);
        v33 = *a1;
        DWORD2(v57) = v59;
        *(_QWORD *)&v57 = v58;
        v12 = sub_3402600(v33, &v54, v10, v48, &v57);
        sub_969240((__int64 *)&v57);
        goto LABEL_5;
      }
      v12 = 0;
      goto LABEL_5;
    }
    v20 = a1[1];
LABEL_21:
    v22 = 1;
    if ( v17 != 1 )
    {
      if ( !v17 )
        goto LABEL_37;
      v22 = v17;
      if ( !*(_QWORD *)(v20 + 8LL * v17 + 112) )
        goto LABEL_37;
    }
    if ( *(_BYTE *)(v20 + 500LL * v22 + 6588) )
      goto LABEL_37;
    goto LABEL_25;
  }
  v27 = sub_3406EB0(*a1, 175, (unsigned int)&v54, v19, v47, (unsigned int)&v58, v56, v57);
LABEL_30:
  v12 = v27;
  if ( !v27 )
    goto LABEL_37;
LABEL_5:
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
  return v12;
}
