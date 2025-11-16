// Function: sub_32BD400
// Address: 0x32bd400
//
__int64 __fastcall sub_32BD400(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __m128i v5; // xmm0
  int v6; // ebx
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  unsigned __int16 v9; // cx
  __int64 v10; // r15
  int v11; // ebx
  __int64 v12; // rdi
  __m128i v13; // xmm1
  __int64 v14; // rax
  int v15; // r9d
  __int64 v16; // r12
  int v18; // eax
  __int64 v19; // rax
  unsigned int *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int128 v30; // rax
  int v31; // r9d
  __int64 v32; // rax
  __int64 v33; // rax
  __int16 v34; // dx
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rsi
  __int128 *v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  int v42; // r9d
  bool v43; // al
  __m128i v44; // [rsp+10h] [rbp-90h] BYREF
  __int128 v45; // [rsp+20h] [rbp-80h]
  unsigned __int16 v46; // [rsp+36h] [rbp-6Ah]
  __int64 v47; // [rsp+38h] [rbp-68h]
  __int64 v48; // [rsp+40h] [rbp-60h] BYREF
  int v49; // [rsp+48h] [rbp-58h]
  __int64 v50; // [rsp+50h] [rbp-50h] BYREF
  __int64 v51; // [rsp+58h] [rbp-48h]
  __m128i v52; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v47 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v44 = v5;
  LODWORD(v45) = v6;
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v48 = v8;
  v46 = v9;
  v11 = v9;
  if ( v8 )
    sub_B96E90((__int64)&v48, v8, 1);
  v12 = *(_QWORD *)a1;
  v13 = _mm_load_si128(&v44);
  v49 = *(_DWORD *)(a2 + 72);
  v52 = v13;
  v50 = v47;
  LODWORD(v51) = v45;
  v14 = sub_3402EA0(v12, 230, (unsigned int)&v48, v11, v10, 0, (__int64)&v50, 2);
  if ( v14 )
  {
    v16 = v14;
    goto LABEL_5;
  }
  v18 = *(_DWORD *)(v47 + 24);
  if ( v18 == 233 )
  {
    v20 = *(unsigned int **)(v47 + 40);
    v21 = *(_QWORD *)(*(_QWORD *)v20 + 48LL) + 16LL * v20[2];
    if ( *(_WORD *)v21 != v46 || *(_QWORD *)(v21 + 8) != v10 && !v46 )
      goto LABEL_12;
    v16 = *(_QWORD *)v20;
    goto LABEL_5;
  }
  if ( v18 != 230 )
  {
    if ( v18 != 152 )
      goto LABEL_12;
    v19 = *(_QWORD *)(v47 + 56);
    if ( !v19 || *(_QWORD *)(v19 + 32) )
      goto LABEL_12;
    v33 = *(_QWORD *)(v47 + 48) + 16LL * (unsigned int)v45;
    v34 = *(_WORD *)v33;
    v35 = *(_QWORD *)(v33 + 8);
    LOWORD(v50) = v34;
    v51 = v35;
    if ( v34 == v46 )
    {
      if ( v46 || v10 == v35 )
      {
LABEL_39:
        v36 = *(_QWORD *)a1;
        v37 = *(_QWORD *)(v47 + 80);
        v38 = *(__int128 **)(v47 + 40);
        v50 = v37;
        if ( v37 )
        {
          *(_QWORD *)&v45 = v38;
          sub_B96E90((__int64)&v50, v37, 1);
          v38 = (__int128 *)v45;
        }
        LODWORD(v51) = *(_DWORD *)(v47 + 72);
        *(_QWORD *)&v45 = sub_3406EB0(v36, 230, (unsigned int)&v50, v11, v10, v15, *v38, *(_OWORD *)&v44);
        *((_QWORD *)&v45 + 1) = v41;
        if ( v50 )
          sub_B91220((__int64)&v50, v50);
        sub_32B3E80(a1, v45, 1, 0, v39, v40);
        v16 = sub_3406EB0(
                *(_QWORD *)a1,
                152,
                (unsigned int)&v48,
                v11,
                v10,
                v42,
                v45,
                *(_OWORD *)(*(_QWORD *)(v47 + 40) + 40LL));
        goto LABEL_5;
      }
    }
    else
    {
      if ( v34 == 15 )
      {
LABEL_12:
        v16 = sub_327CF00((__int64 *)a1, a2);
        goto LABEL_5;
      }
      if ( v34 )
      {
        v43 = (unsigned __int16)(v34 - 17) <= 0xD3u;
        goto LABEL_47;
      }
    }
    v43 = sub_30070B0((__int64)&v50);
LABEL_47:
    if ( v43 )
      goto LABEL_12;
    goto LABEL_39;
  }
  v22 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v23 = *(_QWORD **)(v22 + 24);
  if ( *(_DWORD *)(v22 + 32) > 0x40u )
    v23 = (_QWORD *)*v23;
  v24 = *(_QWORD *)(v47 + 40);
  v25 = *(_QWORD *)(*(_QWORD *)(v24 + 40) + 96LL);
  v26 = *(_QWORD **)(v25 + 24);
  if ( *(_DWORD *)(v25 + 32) > 0x40u )
    v26 = (_QWORD *)*v26;
  v27 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(a1 + 33) )
  {
    v32 = 1;
    if ( v46 != 1 )
    {
      if ( !v46 )
        goto LABEL_35;
      v32 = v46;
      if ( !*(_QWORD *)(v27 + 8LL * v46 + 112) )
        goto LABEL_35;
    }
    if ( *(_BYTE *)(v27 + 500 * v32 + 6644) )
      goto LABEL_35;
  }
  else
  {
    v28 = 1;
    if ( v46 != 1 )
    {
      if ( !v46 )
        goto LABEL_35;
      v28 = v46;
      if ( !*(_QWORD *)(v27 + 8LL * v46 + 112) )
        goto LABEL_35;
    }
    if ( (*(_BYTE *)(v27 + 500 * v28 + 6644) & 0xFB) != 0 )
      goto LABEL_35;
  }
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v24 + 48LL) + 16LL * *(unsigned int *)(v24 + 8)) == 14 && v46 == 11 )
  {
LABEL_35:
    v16 = 0;
    goto LABEL_5;
  }
  v29 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(**(_QWORD **)a1 + 864LL) & 1) == 0 && v26 != (_QWORD *)1 )
    goto LABEL_12;
  *(_QWORD *)&v45 = *(_QWORD *)a1;
  *(_QWORD *)&v30 = sub_3400D50(v29, (v23 == (_QWORD *)1) & (unsigned __int8)(v26 == (_QWORD *)1), &v48, 1);
  v16 = sub_3406EB0(v45, 230, (unsigned int)&v48, v11, v10, v31, *(_OWORD *)*(_QWORD *)(v47 + 40), v30);
LABEL_5:
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v16;
}
