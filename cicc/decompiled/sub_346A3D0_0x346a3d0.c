// Function: sub_346A3D0
// Address: 0x346a3d0
//
void __fastcall sub_346A3D0(__int64 a1, __int64 a2, unsigned __int8 **a3, unsigned __int8 **a4, __int64 a5)
{
  __int64 v8; // rsi
  int v9; // edi
  __int64 v10; // rax
  __int128 v11; // xmm0
  __int64 v12; // r15
  int v13; // edx
  unsigned __int16 *v14; // rax
  __int64 (__fastcall *v15)(__int64, __int64, __int64, __int64, __int64); // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // r15
  unsigned int v21; // esi
  unsigned __int16 v22; // ax
  __int64 v23; // r8
  unsigned __int8 *v24; // r14
  unsigned __int8 *v25; // r15
  __int64 v26; // rdx
  __int128 v27; // rax
  __int64 v28; // r9
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // r15
  __int128 v32; // rax
  __int64 v33; // r9
  __int128 v34; // rax
  __int64 v35; // r9
  unsigned __int8 *v36; // rax
  unsigned int v37; // edx
  unsigned __int8 **v38; // rbx
  int v39; // edx
  int v40; // eax
  __int64 v41; // rsi
  unsigned __int8 *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r15
  __int128 v45; // rax
  __int64 v46; // rax
  unsigned int v47; // edx
  int v48; // edx
  __int128 v49; // [rsp-40h] [rbp-130h]
  __int128 v50; // [rsp-20h] [rbp-110h]
  __int128 v51; // [rsp-10h] [rbp-100h]
  __int64 v52; // [rsp+8h] [rbp-E8h]
  __int128 v53; // [rsp+10h] [rbp-E0h]
  __int64 v54; // [rsp+20h] [rbp-D0h]
  __int128 v55; // [rsp+30h] [rbp-C0h]
  __int64 v56; // [rsp+40h] [rbp-B0h]
  __int128 v58; // [rsp+50h] [rbp-A0h]
  __int128 v59; // [rsp+60h] [rbp-90h]
  __int64 v61; // [rsp+78h] [rbp-78h]
  unsigned int v62; // [rsp+78h] [rbp-78h]
  __int64 v63; // [rsp+B0h] [rbp-40h] BYREF
  int v64; // [rsp+B8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v63 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v63, v8, 1);
  v9 = *(_DWORD *)(a2 + 24);
  v64 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 40);
  v11 = (__int128)_mm_loadu_si128((const __m128i *)v10);
  v12 = 16LL * *(unsigned int *)(v10 + 8);
  v52 = *(_QWORD *)v10;
  v58 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  *a3 = sub_3406EB0(
          (_QWORD *)a5,
          (unsigned int)(v9 != 76) + 56,
          (__int64)&v63,
          *(unsigned __int16 *)(v12 + *(_QWORD *)(*(_QWORD *)v10 + 48LL)),
          *(_QWORD *)(v12 + *(_QWORD *)(*(_QWORD *)v10 + 48LL) + 8),
          *(_QWORD *)v10,
          v11,
          v58);
  *((_DWORD *)a3 + 2) = v13;
  v14 = *(unsigned __int16 **)(a2 + 48);
  v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 528LL);
  *(_QWORD *)&v53 = v14[8];
  *((_QWORD *)&v53 + 1) = *((_QWORD *)v14 + 3);
  v56 = *((_QWORD *)v14 + 1);
  v54 = *v14;
  v61 = *(_QWORD *)(a5 + 64);
  v16 = sub_2E79000(*(__int64 **)(a5 + 40));
  v62 = v15(a1, v16, v61, v54, v56);
  v18 = v17;
  v19 = 1;
  v20 = *(_QWORD *)(v52 + 48) + v12;
  v21 = 2 * (v9 != 76) + 82;
  v22 = *(_WORD *)v20;
  v23 = *(_QWORD *)(v20 + 8);
  if ( (*(_WORD *)v20 == 1 || v22 && (v19 = v22, *(_QWORD *)(a1 + 8LL * v22 + 112)))
    && !*(_BYTE *)(v21 + 500 * v19 + a1 + 6414) )
  {
    v42 = sub_3406EB0((_QWORD *)a5, v21, (__int64)&v63, v22, v23, v52, v11, v58);
    v44 = v43;
    v59 = *(_OWORD *)a3;
    *(_QWORD *)&v45 = sub_33ED040((_QWORD *)a5, 0x16u);
    *((_QWORD *)&v50 + 1) = v44;
    *(_QWORD *)&v50 = v42;
    v46 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)&v63, v62, v18, *((__int64 *)&v59 + 1), v59, v50, v45);
    v38 = a4;
    *a4 = sub_33FB620(a5, v46, v47, (__int64)&v63, v53, *((__int64 *)&v53 + 1), (__m128i)v11, v53);
    v40 = v48;
  }
  else
  {
    *(_QWORD *)&v55 = sub_3400BD0(a5, 0, (__int64)&v63, v22, v23, 0, (__m128i)v11, 0);
    v24 = *a3;
    v25 = a3[1];
    *((_QWORD *)&v55 + 1) = v26;
    *(_QWORD *)&v27 = sub_33ED040((_QWORD *)a5, 0x14u);
    *((_QWORD *)&v49 + 1) = v25;
    *(_QWORD *)&v49 = v24;
    v29 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)&v63, v62, v18, v28, v49, v11, v27);
    v31 = v30;
    *(_QWORD *)&v32 = sub_33ED040((_QWORD *)a5, 2 * (unsigned int)(v9 == 76) + 18);
    *(_QWORD *)&v34 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)&v63, v62, v18, v33, v58, v55, v32);
    *((_QWORD *)&v51 + 1) = v31;
    *(_QWORD *)&v51 = v29;
    v36 = sub_3406EB0((_QWORD *)a5, 0xBCu, (__int64)&v63, v62, v18, v35, v34, v51);
    v38 = a4;
    *a4 = sub_33FB620(a5, (__int64)v36, v37, (__int64)&v63, v53, *((__int64 *)&v53 + 1), (__m128i)v11, v53);
    v40 = v39;
  }
  v41 = v63;
  *((_DWORD *)v38 + 2) = v40;
  if ( v41 )
    sub_B91220((__int64)&v63, v41);
}
