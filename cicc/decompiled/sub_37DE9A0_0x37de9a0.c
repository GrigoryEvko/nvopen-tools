// Function: sub_37DE9A0
// Address: 0x37de9a0
//
int *__fastcall sub_37DE9A0(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // eax
  unsigned __int64 v13; // rsi
  __int32 v14; // ecx
  __m128i *v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // rbx
  char v18; // dl
  __int64 v19; // r8
  int v20; // esi
  unsigned int v21; // edi
  int *v22; // rax
  int v23; // r9d
  __int64 v24; // rax
  __int64 v26; // rax
  unsigned int v27; // esi
  __int32 *v28; // r9
  __int32 v29; // edx
  __m128i *v30; // rax
  __m128i v31; // xmm1
  __int64 v32; // rsi
  __int32 *v33; // rax
  __int32 *v34; // r10
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rax
  unsigned int v38; // esi
  unsigned int v39; // esi
  unsigned int v40; // eax
  __int64 v41; // rdx
  unsigned int v42; // eax
  int v43; // ecx
  unsigned int v44; // r8d
  int v45; // edx
  int v46; // r11d
  int *v47; // r10
  __int64 v48; // [rsp+0h] [rbp-C0h]
  __int64 v49; // [rsp+8h] [rbp-B8h]
  __int64 v50; // [rsp+8h] [rbp-B8h]
  int v51; // [rsp+14h] [rbp-ACh] BYREF
  int *v52; // [rsp+18h] [rbp-A8h] BYREF
  __m128i v53; // [rsp+20h] [rbp-A0h] BYREF
  char v54; // [rsp+38h] [rbp-88h]
  __int64 v55; // [rsp+40h] [rbp-80h]
  _BYTE v56[8]; // [rsp+48h] [rbp-78h]
  __m128i v57[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v58; // [rsp+70h] [rbp-50h] BYREF
  __m128i v59; // [rsp+78h] [rbp-48h]
  int v60; // [rsp+88h] [rbp-38h]

  v5 = a2 + 56;
  v8 = sub_B10CD0(a2 + 56);
  v9 = *(_BYTE *)(v8 - 16);
  if ( (v9 & 2) != 0 )
  {
    if ( *(_DWORD *)(v8 - 24) != 2 )
    {
LABEL_3:
      v49 = 0;
      goto LABEL_4;
    }
    v26 = *(_QWORD *)(v8 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v26 = v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
  }
  v49 = *(_QWORD *)(v26 + 8);
LABEL_4:
  v48 = sub_2E891C0(a2);
  v53.m128i_i64[0] = sub_2E89170(a2);
  if ( v48 )
    sub_AF47B0((__int64)&v53.m128i_i64[1], *(unsigned __int64 **)(v48 + 16), *(unsigned __int64 **)(v48 + 24));
  else
    v54 = 0;
  v10 = v49;
  v50 = *(_QWORD *)a1;
  v55 = v10;
  v11 = sub_B10CD0(v5);
  v12 = sub_37CCE30(v50, &v53, v11);
  v13 = *(unsigned int *)(a4 + 8);
  v51 = v12;
  if ( !v13 )
  {
    v14 = dword_5051178[0];
    v15 = v57;
    do
    {
      v15->m128i_i32[0] = v14;
      v15 = (__m128i *)((char *)v15 + 4);
    }
    while ( &v58 != (__int64 *)v15 );
    v16 = _mm_loadu_si128(a3);
    v58 = 0;
    v60 = 0;
    v59 = v16;
    goto LABEL_10;
  }
  v28 = *(__int32 **)a4;
  v29 = dword_5051178[0];
  v30 = v57;
  do
  {
    v30->m128i_i32[0] = v29;
    v30 = (__m128i *)((char *)v30 + 4);
  }
  while ( v30 != (__m128i *)&v58 );
  v31 = _mm_loadu_si128(a3);
  v58 = (unsigned int)v13;
  v60 = 1;
  v59 = v31;
  if ( v13 > 8 )
    goto LABEL_38;
  v32 = 4 * v13;
  v33 = v28;
  v34 = &v28[(unsigned __int64)v32 / 4];
  v35 = v32 >> 2;
  v36 = v32 >> 4;
  if ( v32 >> 4 )
  {
    while ( *v33 != v29 )
    {
      if ( v33[1] == v29 )
      {
        ++v33;
        break;
      }
      if ( v33[2] == v29 )
      {
        v33 += 2;
        break;
      }
      if ( v33[3] == v29 )
      {
        v33 += 3;
        break;
      }
      v33 += 4;
      if ( v36 == 1 )
      {
        v35 = v34 - v33;
        goto LABEL_58;
      }
      v36 = 1;
    }
LABEL_32:
    if ( v34 == v33 )
      goto LABEL_33;
LABEL_38:
    v60 = 0;
    LODWORD(v58) = 0;
    goto LABEL_10;
  }
LABEL_58:
  if ( v35 != 2 )
  {
    if ( v35 != 3 )
    {
      if ( v35 != 1 )
        goto LABEL_33;
      goto LABEL_61;
    }
    if ( *v33 == v29 )
      goto LABEL_32;
    ++v33;
  }
  if ( *v33 == v29 )
    goto LABEL_32;
  ++v33;
LABEL_61:
  if ( *v33 == v29 )
    goto LABEL_32;
LABEL_33:
  if ( (unsigned int)v32 < 8 )
  {
    if ( (v32 & 4) != 0 )
    {
      v57[0].m128i_i32[0] = *v28;
      *(_DWORD *)&v56[(unsigned int)v32 + 4] = *(__int32 *)((char *)v28 + (unsigned int)v32 - 4);
    }
    else if ( (_DWORD)v32 )
    {
      v57[0].m128i_i8[0] = *(_BYTE *)v28;
    }
  }
  else
  {
    v37 = (unsigned int)v32;
    v38 = v32 - 1;
    *(_QWORD *)&v56[v37] = *(_QWORD *)((char *)v28 + v37 - 8);
    if ( v38 >= 8 )
    {
      v39 = v38 & 0xFFFFFFF8;
      v40 = 0;
      do
      {
        v41 = v40;
        v40 += 8;
        *(__int64 *)((char *)v57[0].m128i_i64 + v41) = *(_QWORD *)((char *)v28 + v41);
      }
      while ( v40 < v39 );
    }
  }
LABEL_10:
  sub_37DE250(a1 + 8, &v51, v57);
  v17 = sub_B10CD0(v5);
  v18 = *(_BYTE *)(a1 + 688) & 1;
  if ( v18 )
  {
    v19 = a1 + 696;
    v20 = 7;
  }
  else
  {
    v27 = *(_DWORD *)(a1 + 704);
    v19 = *(_QWORD *)(a1 + 696);
    if ( !v27 )
    {
      v42 = *(_DWORD *)(a1 + 688);
      ++*(_QWORD *)(a1 + 680);
      v52 = 0;
      v43 = (v42 >> 1) + 1;
LABEL_40:
      v44 = 3 * v27;
      goto LABEL_41;
    }
    v20 = v27 - 1;
  }
  v21 = v20 & (37 * v51);
  v22 = (int *)(v19 + 16LL * v21);
  v23 = *v22;
  if ( *v22 == v51 )
    goto LABEL_13;
  v46 = 1;
  v47 = 0;
  while ( v23 != -1 )
  {
    if ( !v47 && v23 == -2 )
      v47 = v22;
    v21 = v20 & (v46 + v21);
    v22 = (int *)(v19 + 16LL * v21);
    v23 = *v22;
    if ( v51 == *v22 )
      goto LABEL_13;
    ++v46;
  }
  v44 = 24;
  v27 = 8;
  if ( !v47 )
    v47 = v22;
  v42 = *(_DWORD *)(a1 + 688);
  ++*(_QWORD *)(a1 + 680);
  v52 = v47;
  v43 = (v42 >> 1) + 1;
  if ( !v18 )
  {
    v27 = *(_DWORD *)(a1 + 704);
    goto LABEL_40;
  }
LABEL_41:
  if ( 4 * v43 >= v44 )
  {
    v27 *= 2;
    goto LABEL_47;
  }
  if ( v27 - *(_DWORD *)(a1 + 692) - v43 <= v27 >> 3 )
  {
LABEL_47:
    sub_37C5F80(a1 + 680, v27);
    sub_37BDA60(a1 + 680, &v51, &v52);
    v42 = *(_DWORD *)(a1 + 688);
  }
  *(_DWORD *)(a1 + 688) = (2 * (v42 >> 1) + 2) | v42 & 1;
  v22 = v52;
  if ( *v52 != -1 )
    --*(_DWORD *)(a1 + 692);
  v45 = v51;
  *((_QWORD *)v22 + 1) = 0;
  *v22 = v45;
LABEL_13:
  *((_QWORD *)v22 + 1) = v17;
  v24 = sub_B10CD0(v5);
  return sub_37DE580(a1, (__int64)&v53, v24);
}
