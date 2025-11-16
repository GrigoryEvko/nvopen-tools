// Function: sub_393F300
// Address: 0x393f300
//
__int64 __fastcall sub_393F300(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  char *v7; // r13
  size_t v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r12
  unsigned __int64 v12; // rax
  char *v13; // r13
  size_t v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i *v18; // rdx
  __int64 v19; // rdi
  __m128i v20; // xmm0
  _QWORD *v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // r12d
  _QWORD *v27; // rax
  __m128i *v28; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rcx
  char *v41; // rax
  char *v42; // rdx
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx

  v6 = *(_QWORD *)(a1 + 72);
  v7 = *(char **)(v6 + 8);
  v8 = *(_QWORD *)(v6 + 16) - (_QWORD)v7;
  if ( v8 <= 3 )
    goto LABEL_17;
  if ( *(_DWORD *)v7 != 1734567009 )
  {
    v8 = 4;
LABEL_17:
    v27 = sub_16E8CB0();
    v28 = (__m128i *)v27[3];
    v19 = (__int64)v27;
    if ( v27[2] - (_QWORD)v28 <= 0x15u )
    {
      a2 = (unsigned __int64)"Unexpected file type: ";
      v32 = sub_16E7EE0((__int64)v27, "Unexpected file type: ", 0x16u);
      v24 = *(_QWORD *)(v32 + 24);
      v19 = v32;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4530960);
      a2 = 8250;
      v28[1].m128i_i32[0] = 1701869940;
      v28[1].m128i_i16[2] = 8250;
      *v28 = si128;
      v24 = v27[3] + 22LL;
      v27[3] = v24;
    }
    v30 = *(_QWORD *)(v19 + 16) - v24;
    if ( v8 > v30 )
    {
      a2 = (unsigned __int64)v7;
      v31 = sub_16E7EE0(v19, v7, v8);
      v24 = *(_QWORD *)(v31 + 24);
      v19 = v31;
      v30 = *(_QWORD *)(v31 + 16) - v24;
    }
    else if ( v8 )
    {
      v16 = (unsigned int)v8;
      v33 = 0;
      do
      {
        v34 = v33++;
        a2 = (unsigned __int8)v7[v34];
        *(_BYTE *)(v24 + v34) = a2;
      }
      while ( v33 < (unsigned int)v8 );
      v35 = *(_QWORD *)(v19 + 16);
      v24 = v8 + *(_QWORD *)(v19 + 24);
      *(_QWORD *)(v19 + 24) = v24;
      v30 = v35 - v24;
    }
    if ( v30 > 1 )
    {
      v22 = 2606;
      *(_WORD *)v24 = 2606;
      *(_QWORD *)(v19 + 24) += 2LL;
      goto LABEL_14;
    }
LABEL_23:
    a2 = (unsigned __int64)".\n";
    sub_16E7EE0(v19, ".\n", 2u);
    goto LABEL_14;
  }
  *(_QWORD *)(a1 + 80) = 4;
  v9 = *(_QWORD *)(v6 + 8);
  v10 = 4;
  v11 = 8;
  v12 = *(_QWORD *)(v6 + 16) - v9;
  if ( v12 <= 4 )
    v10 = v12;
  if ( v12 <= 8 )
    v11 = v12;
  v13 = (char *)(v9 + v10);
  v14 = v11 - v10;
  if ( v14 != 4 )
    goto LABEL_8;
  if ( *(_DWORD *)v13 != 875573802 && *(_DWORD *)v13 != 875574314 )
  {
    if ( *(_DWORD *)v13 == 875575082 )
    {
      *(_QWORD *)(a1 + 80) = 8;
      v25 = sub_393F200(a1, a2, v10, v9, a5, a6);
      if ( !v25 )
      {
        sub_393D180(a1, a2, v36, v37, v38, v39);
        return 0;
      }
      return v25;
    }
LABEL_8:
    v15 = sub_16E8CB0();
    v18 = (__m128i *)v15[3];
    v19 = (__int64)v15;
    if ( v15[2] - (_QWORD)v18 <= 0x13u )
    {
      a2 = (unsigned __int64)"Unexpected version: ";
      v19 = sub_16E7EE0((__int64)v15, "Unexpected version: ", 0x14u);
      v21 = *(_QWORD **)(v19 + 24);
    }
    else
    {
      v20 = _mm_load_si128((const __m128i *)&xmmword_4530970);
      v18[1].m128i_i32[0] = 540700271;
      *v18 = v20;
      v21 = (_QWORD *)(v15[3] + 20LL);
      *(_QWORD *)(v19 + 24) = v21;
    }
    v22 = *(_QWORD *)(v19 + 16);
    v23 = v22 - (_QWORD)v21;
    if ( v14 > v22 - (__int64)v21 )
    {
      a2 = (unsigned __int64)v13;
      v19 = sub_16E7EE0(v19, v13, v14);
      v21 = *(_QWORD **)(v19 + 24);
      v23 = *(_QWORD *)(v19 + 16) - (_QWORD)v21;
      goto LABEL_12;
    }
    if ( !v14 )
      goto LABEL_12;
    if ( v14 >= 8 )
    {
      a2 = (unsigned __int64)(v21 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *v21 = *(_QWORD *)v13;
      *(_QWORD *)((char *)v21 + v14 - 8) = *(_QWORD *)&v13[v14 - 8];
      v41 = (char *)v21 - a2;
      v42 = (char *)(v13 - v41);
      v43 = (unsigned __int64)&v41[v14] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v43 >= 8 )
      {
        v44 = v43 & 0xFFFFFFFFFFFFFFF8LL;
        v45 = 0;
        do
        {
          v16 = *(_QWORD *)&v42[v45];
          *(_QWORD *)(a2 + v45) = v16;
          v45 += 8LL;
        }
        while ( v45 < v44 );
      }
    }
    else
    {
      if ( (v14 & 4) != 0 )
      {
        *(_DWORD *)v21 = *(_DWORD *)v13;
        *(_DWORD *)((char *)v21 + v14 - 4) = *(_DWORD *)&v13[v14 - 4];
        v40 = *(_QWORD *)(v19 + 16);
        goto LABEL_41;
      }
      *(_BYTE *)v21 = *v13;
      if ( (v14 & 2) != 0 )
      {
        *(_WORD *)((char *)v21 + v14 - 2) = *(_WORD *)&v13[v14 - 2];
        v40 = *(_QWORD *)(v19 + 16);
        goto LABEL_41;
      }
    }
    v40 = *(_QWORD *)(v19 + 16);
LABEL_41:
    v21 = (_QWORD *)(v14 + *(_QWORD *)(v19 + 24));
    v22 = v40 - (_QWORD)v21;
    *(_QWORD *)(v19 + 24) = v21;
    v23 = v22;
LABEL_12:
    if ( v23 > 1 )
    {
      v24 = 2606;
      *(_WORD *)v21 = 2606;
      *(_QWORD *)(v19 + 24) += 2LL;
LABEL_14:
      sub_393D180(v19, a2, v24, v22, v16, v17);
      return 6;
    }
    goto LABEL_23;
  }
  *(_QWORD *)(a1 + 80) = 8;
  sub_393D180(a1, a2, v10, v9, a5, a6);
  return 2;
}
