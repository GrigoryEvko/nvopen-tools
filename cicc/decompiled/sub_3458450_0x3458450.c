// Function: sub_3458450
// Address: 0x3458450
//
__int64 __fastcall sub_3458450(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // eax
  unsigned __int16 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  __int128 v17; // xmm0
  unsigned __int8 *v18; // r12
  unsigned __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // ebx
  __int64 v23; // rsi
  __int128 v24; // rax
  __int64 v25; // r9
  __int128 v26; // rax
  __int64 v27; // r9
  unsigned int v28; // edx
  __int128 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // r9
  __int64 v33; // r12
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rdx
  __int128 v39; // [rsp-50h] [rbp-120h]
  __int128 v40; // [rsp-40h] [rbp-110h]
  __int128 v41; // [rsp-40h] [rbp-110h]
  __int128 v42; // [rsp-30h] [rbp-100h]
  unsigned int v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  unsigned int v45; // [rsp+1Ch] [rbp-B4h]
  __int128 v46; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v47; // [rsp+40h] [rbp-90h]
  __int64 v48; // [rsp+60h] [rbp-70h] BYREF
  int v49; // [rsp+68h] [rbp-68h]
  unsigned int v50; // [rsp+70h] [rbp-60h] BYREF
  __int64 v51; // [rsp+78h] [rbp-58h]
  unsigned __int16 v52; // [rsp+80h] [rbp-50h] BYREF
  __int64 v53; // [rsp+88h] [rbp-48h]

  v6 = *(_QWORD *)(a2 + 80);
  v48 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v48, v6, 1);
  v7 = (__int64 *)a3[5];
  v49 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v50) = v9;
  v51 = v10;
  v11 = sub_2E79000(v7);
  v12 = v50;
  v13 = sub_2FE6750(a1, v50, v51, v11);
  v14 = v50;
  v43 = v13;
  v15 = *(_QWORD *)(a2 + 40);
  v44 = v16;
  v17 = (__int128)_mm_loadu_si128((const __m128i *)(v15 + 40));
  v18 = *(unsigned __int8 **)v15;
  v19 = *(_QWORD *)(v15 + 8);
  v46 = (__int128)_mm_loadu_si128((const __m128i *)(v15 + 80));
  if ( !(_WORD)v50 )
  {
    if ( sub_30070B0((__int64)&v50) )
    {
      v14 = sub_3009970((__int64)&v50, v12, v35, v36, v37);
      v53 = v38;
      v52 = v14;
      if ( !v14 )
        goto LABEL_7;
      goto LABEL_16;
    }
    goto LABEL_5;
  }
  if ( (unsigned __int16)(v50 - 17) > 0xD3u )
  {
LABEL_5:
    v20 = v51;
    goto LABEL_6;
  }
  v14 = word_4456580[(unsigned __int16)v50 - 1];
  v20 = 0;
LABEL_6:
  v52 = v14;
  v53 = v20;
  if ( !v14 )
  {
LABEL_7:
    LODWORD(v21) = sub_3007260((__int64)&v52);
    goto LABEL_8;
  }
LABEL_16:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v21 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
LABEL_8:
  v45 = v21;
  if ( (unsigned int)v21 > 1 )
  {
    v22 = 0;
    do
    {
      v23 = 1LL << v22++;
      *(_QWORD *)&v24 = sub_3400BD0((__int64)a3, v23, (__int64)&v48, v43, v44, 0, (__m128i)v17, 0);
      *((_QWORD *)&v39 + 1) = v19;
      *(_QWORD *)&v39 = v18;
      *(_QWORD *)&v26 = sub_33FC130(a3, 398, (__int64)&v48, v50, v51, v25, v39, v24, v17, v46);
      *((_QWORD *)&v40 + 1) = v19;
      *(_QWORD *)&v40 = v18;
      v18 = sub_33FC130(a3, 400, (__int64)&v48, v50, v51, v27, v40, v26, v17, v46);
      v19 = v28 | v19 & 0xFFFFFFFF00000000LL;
    }
    while ( 1 << v22 < v45 );
  }
  *(_QWORD *)&v29 = sub_34015B0((__int64)a3, (__int64)&v48, v50, v51, 0, 0, (__m128i)v17);
  *((_QWORD *)&v41 + 1) = v19;
  *(_QWORD *)&v41 = v18;
  v47 = sub_33FC130(a3, 407, (__int64)&v48, v50, v51, v30, v41, v29, v17, v46);
  *((_QWORD *)&v42 + 1) = v31 | v19 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v42 = v47;
  v33 = sub_340F900(a3, 0x19Fu, (__int64)&v48, v50, v51, v32, v42, v17, v46);
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v33;
}
