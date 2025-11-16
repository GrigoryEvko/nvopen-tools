// Function: sub_32735C0
// Address: 0x32735c0
//
__int64 __fastcall sub_32735C0(__int64 a1, __int64 a2, int a3, int a4)
{
  _DWORD *v5; // rdx
  __int64 v6; // r11
  int v7; // r12d
  __int64 v8; // rax
  int v10; // edx
  int v11; // ecx
  __int64 v12; // rax
  unsigned __int16 *v13; // rax
  __int64 v14; // r13
  int v15; // r14d
  int v17; // r8d
  __int64 v18; // r9
  __int64 v19; // r10
  __int64 v20; // r11
  char v21; // al
  unsigned __int16 v22; // dx
  char v23; // cl
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // r12
  __int128 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int16 v32; // dx
  int v33; // esi
  bool v34; // al
  __int128 v35; // [rsp-20h] [rbp-A0h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  unsigned __int16 v38; // [rsp+1Eh] [rbp-62h]
  __m128i v39; // [rsp+20h] [rbp-60h]
  __int64 v40; // [rsp+20h] [rbp-60h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  __m128i v42; // [rsp+30h] [rbp-50h]
  __int128 v43; // [rsp+30h] [rbp-50h]
  __int16 v44; // [rsp+40h] [rbp-40h] BYREF
  __int64 v45; // [rsp+48h] [rbp-38h]

  v5 = *(_DWORD **)(a1 + 40);
  v6 = *(_QWORD *)v5;
  v7 = *(_DWORD *)(*(_QWORD *)v5 + 24LL);
  if ( (unsigned int)(v7 - 205) > 1 )
    return 0;
  v8 = *(_QWORD *)(v6 + 56);
  if ( !v8 )
    return 0;
  v10 = v5[2];
  v11 = 1;
  do
  {
    while ( *(_DWORD *)(v8 + 8) != v10 )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_10;
    }
    if ( !v11 )
      return 0;
    v12 = *(_QWORD *)(v8 + 32);
    if ( !v12 )
      goto LABEL_11;
    if ( *(_DWORD *)(v12 + 8) == v10 )
      return 0;
    v8 = *(_QWORD *)(v12 + 32);
    v11 = 0;
  }
  while ( v8 );
LABEL_10:
  if ( v11 == 1 )
    return 0;
LABEL_11:
  v13 = *(unsigned __int16 **)(a1 + 48);
  v14 = *(_QWORD *)(v6 + 40);
  v15 = *(_DWORD *)(a1 + 24);
  v42 = _mm_loadu_si128((const __m128i *)(v14 + 40));
  v38 = *v13;
  v37 = *((_QWORD *)v13 + 1);
  v39 = _mm_loadu_si128((const __m128i *)(v14 + 80));
  if ( !(unsigned __int8)sub_3260D10(v42.m128i_i64[0], v42.m128i_i32[2], v15)
    || !(unsigned __int8)sub_3260D10(v39.m128i_i64[0], v39.m128i_i32[2], v15) )
  {
    return 0;
  }
  v21 = 2;
  if ( v15 != 213 )
    v21 = 2 * (v15 == 214) + 1;
  v22 = *(_WORD *)(v19 + 96);
  if ( !v22 )
    return 0;
  if ( !v38 )
    return 0;
  v23 = 4 * v21;
  if ( (((int)*(unsigned __int16 *)(v18 + 2 * (v22 + 274LL * v38 + 71704) + 6) >> (4 * v21)) & 0xF) != 0 )
    return 0;
  v24 = *(unsigned __int16 *)(*(_QWORD *)(v14 + 80) + 96LL);
  if ( !(_WORD)v24
    || (((int)*(unsigned __int16 *)(v18 + 2 * (274LL * v38 + v24 + 71704) + 6) >> v23) & 0xF) != 0
    || v7 == 206 && v17 > 0 && *(_BYTE *)(v18 + 500LL * v38 + 6620) )
  {
    return 0;
  }
  v36 = v20;
  v25 = sub_33FAF80(a3, v15, a4, v38, v37, v38, *(_OWORD *)&v42);
  v27 = v26;
  v28 = v25;
  *(_QWORD *)&v29 = sub_33FAF80(a3, v15, a4, v38, v37, v38, *(_OWORD *)&v39);
  v43 = v29;
  *(_QWORD *)&v29 = *(_QWORD *)(v36 + 40);
  v30 = *(_QWORD *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  *(_QWORD *)&v29 = *(_QWORD *)(*(_QWORD *)v29 + 48LL) + 16LL * *(unsigned int *)(v29 + 8);
  v32 = *(_WORD *)v29;
  *(_QWORD *)&v29 = *(_QWORD *)(v29 + 8);
  v44 = v32;
  v45 = v29;
  if ( v32 )
  {
    v33 = ((unsigned __int16)(v32 - 17) < 0xD4u) + 205;
  }
  else
  {
    v40 = v30;
    v41 = v31;
    v34 = sub_30070B0((__int64)&v44);
    v31 = v41;
    v30 = v40;
    v33 = 205 - (!v34 - 1);
  }
  *((_QWORD *)&v35 + 1) = v27;
  *(_QWORD *)&v35 = v28;
  return sub_340EC60(a3, v33, a4, v38, v37, 0, v30, v31, v35, v43);
}
