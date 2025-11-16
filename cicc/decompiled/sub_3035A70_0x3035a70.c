// Function: sub_3035A70
// Address: 0x3035a70
//
__int64 __fastcall sub_3035A70(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v9; // rax
  __int64 v11; // rsi
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // rsi
  _QWORD *v16; // rcx
  __int64 v17; // rcx
  int v18; // r13d
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int128 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int128 *v26; // rdx
  __int128 *v27; // rcx
  const __m128i *v28; // r15
  __int64 v29; // r8
  __int64 v30; // r9
  int v31; // ebx
  __int64 v32; // r10
  __int64 v33; // r11
  __int64 v34; // rax
  __int16 v35; // dx
  __int64 v36; // rax
  int v37; // esi
  __int64 v38; // rsi
  _QWORD *v39; // rcx
  bool v40; // al
  __int128 v41; // [rsp-20h] [rbp-E0h]
  __int128 v42; // [rsp-10h] [rbp-D0h]
  __int128 *v43; // [rsp+0h] [rbp-C0h]
  __int64 v44; // [rsp+0h] [rbp-C0h]
  __int64 v45; // [rsp+8h] [rbp-B8h]
  __int128 *v46; // [rsp+10h] [rbp-B0h]
  __int64 v47; // [rsp+10h] [rbp-B0h]
  __int64 v48; // [rsp+18h] [rbp-A8h]
  unsigned __int16 v49; // [rsp+20h] [rbp-A0h]
  __m128i v50; // [rsp+20h] [rbp-A0h]
  __int64 v51; // [rsp+20h] [rbp-A0h]
  __int64 v52; // [rsp+30h] [rbp-90h]
  __int64 v53; // [rsp+30h] [rbp-90h]
  int v54; // [rsp+30h] [rbp-90h]
  __int64 v55; // [rsp+38h] [rbp-88h]
  __int128 v56; // [rsp+40h] [rbp-80h] BYREF
  __int64 v57; // [rsp+50h] [rbp-70h] BYREF
  int v58; // [rsp+58h] [rbp-68h]
  _QWORD v59[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v60; // [rsp+70h] [rbp-50h] BYREF
  int v61; // [rsp+78h] [rbp-48h]
  __int16 v62; // [rsp+80h] [rbp-40h] BYREF
  __int64 v63; // [rsp+88h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 56);
  *(_QWORD *)&v56 = a4;
  *((_QWORD *)&v56 + 1) = a5;
  if ( !v6 || *(_QWORD *)(v6 + 32) || *(_DWORD *)(a2 + 24) != 205 )
    return 0;
  v9 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 != 35 && v12 != 11 )
    goto LABEL_7;
  v38 = *(_QWORD *)(v11 + 96);
  v39 = *(_QWORD **)(v38 + 24);
  if ( *(_DWORD *)(v38 + 32) > 0x40u )
    v39 = (_QWORD *)*v39;
  if ( !v39 )
  {
    v17 = 80;
    v18 = 1;
  }
  else
  {
LABEL_7:
    v13 = *(_QWORD *)(v9 + 80);
    v14 = *(_DWORD *)(v13 + 24);
    if ( v14 != 11 && v14 != 35 )
      return 0;
    v15 = *(_QWORD *)(v13 + 96);
    v16 = *(_QWORD **)(v15 + 24);
    if ( *(_DWORD *)(v15 + 32) > 0x40u )
      v16 = (_QWORD *)*v16;
    if ( v16 )
      return 0;
    v17 = 40;
    v18 = 2;
  }
  v19 = *(_QWORD *)(v9 + v17);
  if ( *(_DWORD *)(v19 + 24) != 58 )
    return 0;
  v20 = *(_QWORD *)(v19 + 56);
  if ( !v20 || *(_QWORD *)(v20 + 32) )
    return 0;
  v21 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v55 = *(_QWORD *)(v21 + 8);
  v49 = *(_WORD *)v21;
  v57 = *(_QWORD *)(a1 + 80);
  if ( v57 )
  {
    v52 = a6;
    sub_2AAAFA0(&v57);
    a6 = v52;
  }
  v22 = *(_QWORD *)(a6 + 16);
  v53 = a6;
  v58 = *(_DWORD *)(a1 + 72);
  *(_QWORD *)&v23 = sub_3406EB0(
                      v22,
                      58,
                      (unsigned int)&v57,
                      v49,
                      v55,
                      a6,
                      *(_OWORD *)*(_QWORD *)(v19 + 40),
                      *(_OWORD *)(*(_QWORD *)(v19 + 40) + 40LL));
  v59[0] = sub_3406EB0(*(_QWORD *)(v53 + 16), 56, (unsigned int)&v57, v49, v55, v53, v23, v56);
  v24 = *(_QWORD *)(v53 + 16);
  v59[1] = v25;
  v54 = v24;
  if ( v18 == 1 )
  {
    v26 = (__int128 *)v59;
    v27 = &v56;
  }
  else
  {
    v26 = &v56;
    v27 = (__int128 *)v59;
  }
  v28 = *(const __m128i **)(a2 + 40);
  v60 = *(_QWORD *)(a1 + 80);
  if ( v60 )
  {
    v43 = v26;
    v46 = v27;
    sub_2AAAFA0(&v60);
    v26 = v43;
    v27 = v46;
  }
  v29 = *(_QWORD *)v26;
  v30 = *((_QWORD *)v26 + 1);
  v31 = v49;
  v32 = *(_QWORD *)v27;
  v33 = *((_QWORD *)v27 + 1);
  v61 = *(_DWORD *)(a1 + 72);
  v34 = *(_QWORD *)(v28->m128i_i64[0] + 48) + 16LL * v28->m128i_u32[2];
  v35 = *(_WORD *)v34;
  v50 = _mm_loadu_si128(v28);
  v36 = *(_QWORD *)(v34 + 8);
  v62 = v35;
  v63 = v36;
  if ( v35 )
  {
    v37 = 206 - ((unsigned __int16)(v35 - 17) >= 0xD4u);
  }
  else
  {
    v44 = v32;
    v45 = v33;
    v47 = v29;
    v48 = v30;
    v40 = sub_30070B0((__int64)&v62);
    v32 = v44;
    v33 = v45;
    v29 = v47;
    v30 = v48;
    v37 = 206 - !v40;
  }
  *((_QWORD *)&v42 + 1) = v30;
  *(_QWORD *)&v42 = v29;
  *((_QWORD *)&v41 + 1) = v33;
  *(_QWORD *)&v41 = v32;
  v51 = sub_340EC60(v54, v37, (unsigned int)&v60, v31, v55, 0, v50.m128i_i64[0], v50.m128i_i64[1], v41, v42);
  sub_9C6650(&v60);
  sub_9C6650(&v57);
  return v51;
}
