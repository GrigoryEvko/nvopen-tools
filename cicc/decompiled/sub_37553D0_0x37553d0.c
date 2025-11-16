// Function: sub_37553D0
// Address: 0x37553d0
//
__int64 __fastcall sub_37553D0(__int64 a1, unsigned __int8 *a2, unsigned int a3, char a4, signed int a5, __m128i *a6)
{
  __int64 v8; // rax
  __int64 v9; // r9
  char v10; // r10
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // ebx
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rbx
  char v17; // r11
  __int64 v18; // rsi
  __int64 v19; // rax
  __int16 v20; // ax
  int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned __int16); // rax
  __int32 v35; // eax
  unsigned __int8 *v36; // rsi
  __int32 v37; // r8d
  signed int v38; // ebx
  __int64 v39; // r14
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int32 v43; // ecx
  __m128i *v44; // rsi
  int v45; // ecx
  int v46; // r8d
  unsigned int v47; // eax
  __int8 *v48; // rdx
  unsigned int v49; // eax
  __m128i *v50; // rdi
  int v51; // edx
  __int32 v52; // edx
  int v53; // esi
  unsigned int i; // eax
  __int8 *v55; // rcx
  unsigned int v56; // eax
  unsigned __int32 v57; // eax
  __int64 v58; // rax
  unsigned __int32 v59; // eax
  __int64 v61; // [rsp+8h] [rbp-C8h]
  char v62; // [rsp+10h] [rbp-C0h]
  __int64 v63; // [rsp+18h] [rbp-B8h]
  char v64; // [rsp+18h] [rbp-B8h]
  __int32 v65; // [rsp+18h] [rbp-B8h]
  unsigned __int16 v67; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v69; // [rsp+30h] [rbp-A0h]
  unsigned __int8 *v71; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int8 *v72; // [rsp+50h] [rbp-80h] BYREF
  __int64 v73; // [rsp+58h] [rbp-78h]
  _QWORD v74[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v75; // [rsp+70h] [rbp-60h] BYREF
  __int64 v76; // [rsp+80h] [rbp-50h]
  __int64 v77; // [rsp+88h] [rbp-48h]
  __int64 v78; // [rsp+90h] [rbp-40h]

  if ( a5 < 0 )
  {
    if ( a4 )
    {
      if ( (a6->m128i_i8[8] & 1) != 0 )
      {
        v50 = a6 + 1;
        v51 = 15;
      }
      else
      {
        v52 = a6[1].m128i_i32[2];
        v50 = (__m128i *)a6[1].m128i_i64[0];
        if ( !v52 )
          goto LABEL_47;
        v51 = v52 - 1;
      }
      v53 = 1;
      for ( i = v51 & (a3 + (((unsigned __int64)a2 >> 9) ^ ((unsigned __int64)a2 >> 4))); ; i = v51 & v56 )
      {
        v55 = &v50->m128i_i8[24 * i];
        if ( a2 == *(unsigned __int8 **)v55 && a3 == *((_DWORD *)v55 + 2) )
          break;
        if ( !*(_QWORD *)v55 && *((_DWORD *)v55 + 2) == -1 )
          goto LABEL_47;
        v56 = v53 + i;
        ++v53;
      }
      *(_QWORD *)v55 = 0;
      *((_DWORD *)v55 + 2) = -2;
      v59 = a6->m128i_u32[2];
      ++a6->m128i_i32[3];
      a6->m128i_i32[2] = (2 * (v59 >> 1) - 2) | v59 & 1;
    }
LABEL_47:
    v72 = a2;
    LODWORD(v73) = a3;
    LODWORD(v74[0]) = a5;
    return sub_3755010((__int64)&v75, a6, (unsigned __int64 *)&v72, v74);
  }
  v8 = *(unsigned __int16 *)(*((_QWORD *)a2 + 6) + 16LL * a3);
  v67 = v8;
  if ( (_WORD)v8 && (v33 = *(_QWORD *)(a1 + 32), (v69 = *(_QWORD *)(v33 + 8 * v8 + 112)) != 0) )
  {
    v34 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v33 + 552LL);
    if ( v34 == sub_2EC09E0 )
    {
      v9 = *((_QWORD *)a2 + 7);
      if ( !v9 )
      {
        v28 = sub_2FF6500(*(_QWORD *)(a1 + 24), a5, v67);
        goto LABEL_28;
      }
      goto LABEL_4;
    }
    v58 = ((__int64 (__fastcall *)(__int64, _QWORD, bool))v34)(v33, v67, (a2[32] & 4) != 0);
    v9 = *((_QWORD *)a2 + 7);
    v69 = v58;
    if ( !v9 )
    {
      v28 = sub_2FF6500(*(_QWORD *)(a1 + 24), a5, v67);
      if ( v69 )
        goto LABEL_28;
      goto LABEL_71;
    }
  }
  else
  {
    v69 = 0;
    v9 = *((_QWORD *)a2 + 7);
    if ( !v9 )
    {
      v28 = sub_2FF6500(*(_QWORD *)(a1 + 24), a5, v8);
LABEL_71:
      v69 = (unsigned __int64)v28;
      goto LABEL_28;
    }
  }
LABEL_4:
  v10 = 1;
  v11 = *(_QWORD *)(a1 + 24);
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v9 + 16);
      if ( *(_DWORD *)(v14 + 24) == 49 )
      {
        v12 = *(_QWORD *)(v14 + 40);
        if ( a2 == *(unsigned __int8 **)(v12 + 80) && a3 == *(_DWORD *)(v12 + 88) )
        {
          v13 = *(_DWORD *)(*(_QWORD *)(v12 + 40) + 96LL);
          if ( v13 >= 0 )
          {
            if ( a5 != v13 )
              v10 = 0;
            goto LABEL_10;
          }
          sub_2FF6500(v11, a5, v67);
          v32 = *(_QWORD *)(a1 + 8);
          v69 = *(_QWORD *)(*(_QWORD *)(v32 + 56) + 16LL * (v13 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_36:
          v35 = sub_2EC06C0(v32, v69, byte_3F871B3, 0, v29, v30);
          v36 = (unsigned __int8 *)*((_QWORD *)a2 + 10);
          v37 = v35;
          v38 = v35;
          v71 = v36;
          v39 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) - 800LL;
          if ( v36 )
          {
            v65 = v35;
            sub_B96E90((__int64)&v71, (__int64)v36, 1);
            v37 = v65;
            v72 = v71;
            if ( v71 )
            {
              sub_B976B0((__int64)&v71, v71, (__int64)&v72);
              v37 = v65;
              v71 = 0;
            }
          }
          else
          {
            v72 = 0;
          }
          v73 = 0;
          v74[0] = 0;
          v40 = sub_2F26260(*(_QWORD *)(a1 + 40), *(__int64 **)(a1 + 48), (__int64 *)&v72, v39, v37);
          v75.m128i_i64[0] = 0;
          v76 = 0;
          v75.m128i_i32[2] = a5;
          v77 = 0;
          v78 = 0;
          sub_2E8EAD0(v41, (__int64)v40, &v75);
          if ( v72 )
            sub_B91220((__int64)&v72, (__int64)v72);
          if ( v71 )
            sub_B91220((__int64)&v71, (__int64)v71);
          goto LABEL_43;
        }
      }
      v15 = *(unsigned int *)(v14 + 64);
      v16 = 0;
      v17 = 1;
      if ( (_DWORD)v15 )
        break;
LABEL_10:
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        goto LABEL_25;
    }
    do
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(v14 + 40) + 40 * v16;
        v19 = *(unsigned int *)(v18 + 8);
        if ( *(unsigned __int8 **)v18 != a2 )
          goto LABEL_14;
        if ( a3 != (_DWORD)v19 )
          goto LABEL_14;
        v20 = *(_WORD *)(*((_QWORD *)a2 + 6) + 16 * v19);
        if ( v20 == 1 || v20 == 262 )
          goto LABEL_14;
        v21 = *(_DWORD *)(v14 + 24);
        v17 = 0;
        if ( v21 >= 0 )
          goto LABEL_14;
        v22 = *(_QWORD **)(a1 + 16);
        v23 = v22[1] - 40LL * (unsigned int)~v21;
        v24 = *(unsigned __int8 *)(v23 + 4) + (unsigned int)v16;
        if ( (unsigned int)v24 >= *(unsigned __int16 *)(v23 + 2) )
          goto LABEL_14;
        v61 = v9;
        v62 = v10;
        v63 = v11;
        v25 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(*v22 + 16LL))(
                          v22,
                          v23,
                          v24,
                          v11,
                          *(_QWORD *)a1);
        v26 = sub_2FF6410(v63, v25);
        v17 = 0;
        v10 = v62;
        v9 = v61;
        if ( !v69 )
        {
          v69 = (unsigned __int64)v26;
          v11 = *(_QWORD *)(a1 + 24);
          goto LABEL_14;
        }
        v11 = *(_QWORD *)(a1 + 24);
        if ( v26 )
          break;
LABEL_14:
        if ( v15 == ++v16 )
          goto LABEL_24;
      }
      v27 = sub_2FF6970(*(_QWORD *)(a1 + 24), v69, (__int64)v26);
      v17 = 0;
      v10 = v62;
      v9 = v61;
      if ( !v27 )
      {
        v11 = *(_QWORD *)(a1 + 24);
        goto LABEL_14;
      }
      ++v16;
      v69 = v27;
      v11 = *(_QWORD *)(a1 + 24);
    }
    while ( v15 != v16 );
LABEL_24:
    v9 = *(_QWORD *)(v9 + 32);
    v10 &= v17;
  }
  while ( v9 );
LABEL_25:
  v64 = v10;
  v28 = sub_2FF6500(v11, a5, v67);
  v31 = v69;
  if ( !v69 )
    v31 = (unsigned __int64)v28;
  v69 = v31;
  if ( !v64 )
  {
LABEL_29:
    v32 = *(_QWORD *)(a1 + 8);
    goto LABEL_36;
  }
LABEL_28:
  if ( *(char *)(*v28 + 28) >= 0 )
    goto LABEL_29;
  v38 = a5;
LABEL_43:
  if ( !a4 )
    goto LABEL_44;
  if ( (a6->m128i_i8[8] & 1) != 0 )
  {
    v44 = a6 + 1;
    v45 = 15;
  }
  else
  {
    v43 = a6[1].m128i_i32[2];
    v44 = (__m128i *)a6[1].m128i_i64[0];
    if ( !v43 )
      goto LABEL_44;
    v45 = v43 - 1;
  }
  v46 = 1;
  v47 = v45 & (a3 + (((unsigned __int64)a2 >> 9) ^ ((unsigned __int64)a2 >> 4)));
  while ( 2 )
  {
    v48 = &v44->m128i_i8[24 * v47];
    if ( a2 != *(unsigned __int8 **)v48 )
    {
      if ( !*(_QWORD *)v48 && *((_DWORD *)v48 + 2) == -1 )
        goto LABEL_44;
      goto LABEL_56;
    }
    if ( a3 != *((_DWORD *)v48 + 2) )
    {
LABEL_56:
      v49 = v46 + v47;
      ++v46;
      v47 = v45 & v49;
      continue;
    }
    break;
  }
  *(_QWORD *)v48 = 0;
  *((_DWORD *)v48 + 2) = -2;
  v57 = a6->m128i_u32[2];
  ++a6->m128i_i32[3];
  a6->m128i_i32[2] = (2 * (v57 >> 1) - 2) | v57 & 1;
LABEL_44:
  LODWORD(v74[0]) = v38;
  v72 = a2;
  LODWORD(v73) = a3;
  return sub_3755010((__int64)&v75, a6, (unsigned __int64 *)&v72, v74);
}
