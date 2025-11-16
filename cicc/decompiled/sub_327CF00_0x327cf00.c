// Function: sub_327CF00
// Address: 0x327cf00
//
__int64 __fastcall sub_327CF00(__int64 *a1, __int64 a2)
{
  unsigned __int16 *v2; // rax
  int v3; // r14d
  unsigned __int16 v4; // r13
  __int64 v5; // r15
  unsigned int v8; // r9d
  _DWORD *v10; // rdx
  __int64 v11; // r10
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  const __m128i *v15; // rdx
  unsigned __int16 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r10
  unsigned int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int128 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // edx
  int v31; // r9d
  unsigned int v32; // edx
  int v33; // r9d
  char v34; // r13
  unsigned int v35; // edx
  int v36; // r9d
  __int128 v37; // [rsp-10h] [rbp-130h]
  __int128 v38; // [rsp+0h] [rbp-120h]
  unsigned int v39; // [rsp+14h] [rbp-10Ch]
  __int64 v40; // [rsp+18h] [rbp-108h]
  __int64 v41; // [rsp+20h] [rbp-100h]
  __int128 v42; // [rsp+20h] [rbp-100h]
  __int64 v43; // [rsp+30h] [rbp-F0h]
  __int128 v44; // [rsp+30h] [rbp-F0h]
  __int64 (__fastcall *v45)(__int64, __int64, __int64, __int64, __int64); // [rsp+40h] [rbp-E0h]
  __int64 v46; // [rsp+40h] [rbp-E0h]
  __int64 v47; // [rsp+40h] [rbp-E0h]
  __int64 v48; // [rsp+50h] [rbp-D0h]
  __int64 v49; // [rsp+50h] [rbp-D0h]
  __int128 v50; // [rsp+50h] [rbp-D0h]
  unsigned __int16 v51; // [rsp+A0h] [rbp-80h]
  __int64 v52; // [rsp+A8h] [rbp-78h]
  __int16 v53; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v54; // [rsp+B8h] [rbp-68h]
  __int64 v55; // [rsp+C0h] [rbp-60h] BYREF
  int v56; // [rsp+C8h] [rbp-58h]
  __int64 v57; // [rsp+D0h] [rbp-50h]
  __int64 v58; // [rsp+D8h] [rbp-48h]

  v2 = *(unsigned __int16 **)(a2 + 48);
  v3 = *(_DWORD *)(a2 + 24);
  v4 = *v2;
  v51 = *v2;
  v52 = *((_QWORD *)v2 + 1);
  if ( *((_BYTE *)a1 + 33) )
    return 0;
  v5 = a1[1];
  v8 = 1;
  if ( v4 != 1 )
  {
    if ( !v4 )
      return 0;
    v8 = v4;
    if ( !*(_QWORD *)(v5 + 8LL * v4 + 112) )
      return 0;
  }
  if ( (*(_BYTE *)(v5 + 500LL * v8 + 6620) & 0xFB) != 0 )
    return 0;
  v10 = *(_DWORD **)(a2 + 40);
  v11 = *(_QWORD *)v10;
  if ( *(_DWORD *)(*(_QWORD *)v10 + 24LL) != 206 )
    return 0;
  v12 = *(_QWORD *)(v11 + 56);
  if ( !v12 )
    return 0;
  v13 = v10[2];
  v14 = 1;
  do
  {
    if ( v13 == *(_DWORD *)(v12 + 8) )
    {
      if ( !v14 )
        return 0;
      v12 = *(_QWORD *)(v12 + 32);
      if ( !v12 )
        goto LABEL_18;
      if ( v13 == *(_DWORD *)(v12 + 8) )
        return 0;
      v14 = 0;
    }
    v12 = *(_QWORD *)(v12 + 32);
  }
  while ( v12 );
  if ( v14 == 1 )
    return 0;
LABEL_18:
  v15 = *(const __m128i **)(v11 + 40);
  if ( *(_DWORD *)(v15->m128i_i64[0] + 24) != 208 )
    return 0;
  v39 = v8;
  v40 = v11;
  v38 = (__int128)_mm_loadu_si128(v15);
  v16 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v15->m128i_i64[0] + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v15->m128i_i64[0] + 40) + 8LL));
  v48 = *(_QWORD *)(*a1 + 64);
  v41 = *((_QWORD *)v16 + 1);
  v43 = *v16;
  v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v5 + 528LL);
  v17 = sub_2E79000(*(__int64 **)(*a1 + 40));
  LOWORD(v18) = v45(v5, v17, v48, v43, v41);
  v19 = v40;
  v20 = v39;
  v53 = v18;
  v54 = v21;
  if ( v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_36;
  v23 = *(_QWORD *)&byte_444C4A0[16 * (v39 - 1)];
  v34 = byte_444C4A0[16 * (v39 - 1) + 8];
  if ( (_WORD)v18 )
  {
    if ( (_WORD)v18 != 1 && (unsigned __int16)(v18 - 504) > 7u )
    {
      v26 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v18 - 16];
      LOBYTE(v18) = byte_444C4A0[16 * (unsigned __int16)v18 - 8];
      goto LABEL_22;
    }
LABEL_36:
    BUG();
  }
  v49 = *(_QWORD *)&byte_444C4A0[16 * (v39 - 1)];
  v22 = sub_3007260((__int64)&v53);
  v19 = v40;
  v23 = v49;
  v24 = v22;
  v18 = v25;
  v57 = v24;
  v26 = v24;
  v58 = v18;
LABEL_22:
  if ( v23 != v26 || (_BYTE)v18 != v34 )
    return 0;
  v27 = *(_QWORD *)(v19 + 40);
  v28 = *(_OWORD *)(v27 + 40);
  v44 = (__int128)_mm_loadu_si128((const __m128i *)(v27 + 80));
  v55 = *(_QWORD *)(a2 + 80);
  if ( v55 )
  {
    v42 = v28;
    sub_325F5D0(&v55);
    v28 = v42;
  }
  v29 = *a1;
  v56 = *(_DWORD *)(a2 + 72);
  if ( v3 == 230 )
  {
    *(_QWORD *)&v50 = sub_3406EB0(
                        v29,
                        230,
                        (unsigned int)&v55,
                        v51,
                        v52,
                        v20,
                        v28,
                        *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
    *((_QWORD *)&v50 + 1) = v35;
    v46 = sub_3406EB0(*a1, 230, (unsigned int)&v55, v51, v52, v36, v44, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  }
  else
  {
    *(_QWORD *)&v50 = sub_33FAF80(v29, v3, (unsigned int)&v55, v51, v52, v20, v28);
    *((_QWORD *)&v50 + 1) = v30;
    v46 = sub_33FAF80(*a1, v3, (unsigned int)&v55, v51, v52, v31, v44);
  }
  *((_QWORD *)&v37 + 1) = v32;
  *(_QWORD *)&v37 = v46;
  v47 = sub_340F900(*a1, 206, (unsigned int)&v55, v51, v52, v33, v38, v50, v37);
  sub_9C6650(&v55);
  return v47;
}
