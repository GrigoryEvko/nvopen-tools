// Function: sub_38443E0
// Address: 0x38443e0
//
__int64 __fastcall sub_38443E0(__int64 a1, __int64 a2)
{
  const __m128i *v4; // rax
  int *v5; // r12
  char v6; // si
  __int64 v7; // r9
  int v8; // ecx
  unsigned int v9; // edi
  __int64 v10; // rax
  int v11; // r10d
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // r11
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int16 *v22; // rax
  __int64 v23; // r8
  unsigned int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned __int16 *v31; // rax
  __int64 v32; // r8
  __int64 v33; // rcx
  unsigned __int8 *v34; // rax
  __int64 v35; // rax
  int v36; // eax
  int v37; // r8d
  __int128 v38; // [rsp-30h] [rbp-C0h]
  __int128 v39; // [rsp-20h] [rbp-B0h]
  __int128 v40; // [rsp-20h] [rbp-B0h]
  __int128 v41; // [rsp-10h] [rbp-A0h]
  __int64 v42; // [rsp+8h] [rbp-88h]
  unsigned int v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+10h] [rbp-80h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+20h] [rbp-70h]
  __int64 v47; // [rsp+20h] [rbp-70h]
  __int64 v48; // [rsp+28h] [rbp-68h]
  __int64 v49; // [rsp+30h] [rbp-60h]
  __int64 v50; // [rsp+30h] [rbp-60h]
  unsigned int v51; // [rsp+30h] [rbp-60h]
  __int64 v52; // [rsp+38h] [rbp-58h]
  unsigned int v53; // [rsp+38h] [rbp-58h]
  __int64 v54; // [rsp+38h] [rbp-58h]
  __int128 v55; // [rsp+40h] [rbp-50h]
  __int64 v56; // [rsp+50h] [rbp-40h] BYREF
  int v57; // [rsp+58h] [rbp-38h]

  v4 = *(const __m128i **)(a2 + 40);
  v55 = (__int128)_mm_loadu_si128(v4);
  LODWORD(v56) = sub_375D5B0(a1, v4[2].m128i_u64[1], v4[3].m128i_i64[0]);
  v5 = sub_3805BC0(a1 + 712, (int *)&v56);
  sub_37593F0(a1, v5);
  v6 = *(_BYTE *)(a1 + 512) & 1;
  if ( v6 )
  {
    v7 = a1 + 520;
    v8 = 7;
  }
  else
  {
    v29 = *(unsigned int *)(a1 + 528);
    v7 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v29 )
      goto LABEL_17;
    v8 = v29 - 1;
  }
  v9 = v8 & (37 * *v5);
  v10 = v7 + 24LL * v9;
  v11 = *(_DWORD *)v10;
  if ( *v5 == *(_DWORD *)v10 )
    goto LABEL_4;
  v36 = 1;
  while ( v11 != -1 )
  {
    v37 = v36 + 1;
    v9 = v8 & (v36 + v9);
    v10 = v7 + 24LL * v9;
    v11 = *(_DWORD *)v10;
    if ( *v5 == *(_DWORD *)v10 )
      goto LABEL_4;
    v36 = v37;
  }
  if ( v6 )
  {
    v35 = 192;
    goto LABEL_18;
  }
  v29 = *(unsigned int *)(a1 + 528);
LABEL_17:
  v35 = 24 * v29;
LABEL_18:
  v10 = v7 + v35;
LABEL_4:
  v49 = *(_QWORD *)(v10 + 8);
  v12 = v49;
  v52 = *(unsigned int *)(v10 + 16);
  v13 = v52;
  v14 = sub_37AE0F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v16 = *(unsigned int *)(a2 + 24);
  v17 = v14;
  v18 = v15;
  v19 = *(_QWORD **)(a1 + 8);
  v20 = 16 * v52;
  if ( (unsigned int)(v16 - 488) <= 1 )
  {
    v30 = *(_QWORD *)(a2 + 80);
    v31 = (unsigned __int16 *)(*(_QWORD *)(v49 + 48) + v20);
    v32 = *((_QWORD *)v31 + 1);
    v33 = *v31;
    v54 = *(_QWORD *)(a2 + 40);
    v56 = v30;
    if ( v30 )
    {
      v42 = v33;
      v44 = v17;
      v45 = v15;
      v47 = v32;
      v51 = v16;
      sub_B96E90((__int64)&v56, v30, 1);
      v33 = v42;
      v17 = v44;
      v18 = v45;
      v32 = v47;
      v16 = v51;
    }
    v57 = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v40 + 1) = v18;
    *(_QWORD *)&v40 = v17;
    *((_QWORD *)&v38 + 1) = v13;
    *(_QWORD *)&v38 = v12;
    v34 = sub_33FC130(v19, (unsigned int)v16, (__int64)&v56, v33, v32, v16, v55, v38, v40, *(_OWORD *)(v54 + 120));
    v26 = v56;
    v27 = (__int64)v34;
    if ( v56 )
      goto LABEL_8;
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 80);
    v22 = (unsigned __int16 *)(*(_QWORD *)(v49 + 48) + v20);
    v23 = *((_QWORD *)v22 + 1);
    v24 = *v22;
    v56 = v21;
    if ( v21 )
    {
      v43 = v24;
      v46 = v17;
      v48 = v15;
      v50 = v23;
      v53 = v16;
      sub_B96E90((__int64)&v56, v21, 1);
      v24 = v43;
      v17 = v46;
      v18 = v48;
      v23 = v50;
      v16 = v53;
    }
    *((_QWORD *)&v41 + 1) = v18;
    *(_QWORD *)&v41 = v17;
    *((_QWORD *)&v39 + 1) = v13;
    *(_QWORD *)&v39 = v12;
    v57 = *(_DWORD *)(a2 + 72);
    v25 = sub_340F900(v19, v16, (__int64)&v56, v24, v23, v16, v55, v39, v41);
    v26 = v56;
    v27 = v25;
    if ( v56 )
LABEL_8:
      sub_B91220((__int64)&v56, v26);
  }
  return v27;
}
