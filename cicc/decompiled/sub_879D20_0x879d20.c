// Function: sub_879D20
// Address: 0x879d20
//
__int64 __fastcall sub_879D20(const __m128i *a1, char a2, __int64 a3, __int128 *a4, int a5, __m128i *a6)
{
  __m128i *v6; // r11
  const __m128i *v7; // r15
  __int64 v8; // rax
  _BOOL4 v9; // r8d
  __int64 v10; // r9
  __int64 v12; // rbx
  __int64 v13; // r11
  char v14; // al
  _QWORD *v15; // r15
  int v16; // r10d
  __int64 v17; // rcx
  __int64 v18; // rdx
  char v19; // si
  __int64 v20; // r12
  __int64 v21; // rdi
  char v22; // cl
  __int64 v23; // rdi
  int v25; // eax
  __int128 *v26; // rdi
  int v27; // eax
  int v28; // eax
  __int64 v29; // rdi
  char v30; // r10
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __m128i *v37; // [rsp+28h] [rbp-48h]
  _BOOL4 v39; // [rsp+38h] [rbp-38h]
  _BOOL4 v40; // [rsp+3Ch] [rbp-34h]
  _BOOL4 v41; // [rsp+3Ch] [rbp-34h]

  v6 = a6;
  v7 = a1;
  *a6 = _mm_loadu_si128(a1);
  a6[1] = _mm_loadu_si128(a1 + 1);
  a6[2] = _mm_loadu_si128(a1 + 2);
  a6[3] = _mm_loadu_si128(a1 + 3);
  if ( (a6[1].m128i_i8[1] & 0x20) != 0 )
    goto LABEL_31;
  v8 = a6->m128i_i64[0];
  v9 = 0;
  v10 = 0;
  if ( dword_4F077C4 == 2 )
  {
    v9 = a2 == 3;
    if ( (a1[1].m128i_i8[0] & 4) == 0 && ((a1[1].m128i_i8[2] & 2) != 0 || (v10 = a1[2].m128i_i64[0]) == 0) )
    {
      v10 = 0;
      if ( dword_4F04C34 )
        v10 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C34 + 184) + 32LL);
    }
  }
  v12 = *(_QWORD *)(v8 + 40);
  if ( !v12 )
  {
LABEL_31:
    v23 = 0;
    goto LABEL_20;
  }
  v37 = v6;
  v39 = 0;
  v13 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(v12 + 40) != unk_4F066A8 )
        goto LABEL_5;
      v14 = *(_BYTE *)(v12 + 80);
      if ( v14 == 14 )
      {
        v15 = *(_QWORD **)(v12 + 88);
        v16 = 1;
        v17 = v15[1];
      }
      else
      {
        if ( v14 != 15 )
          goto LABEL_5;
        v15 = *(_QWORD **)(v12 + 88);
        v16 = 0;
        v17 = v15[1];
      }
      v18 = *v15;
      v19 = *(_BYTE *)(*v15 + 140LL);
      v20 = *v15;
      if ( v19 == 12 )
      {
        do
          v20 = *(_QWORD *)(v20 + 160);
        while ( *(_BYTE *)(v20 + 140) == 12 );
      }
      v21 = *(_QWORD *)(v12 + 64);
      v22 = *(_BYTE *)(v17 + 88) & 0x70;
      if ( v9 )
        break;
      if ( !(a3 | v10) && v22 == 48 )
        goto LABEL_19;
LABEL_16:
      if ( v10 != v21 )
        goto LABEL_5;
      if ( v14 == 14 )
      {
        if ( !a3 )
          goto LABEL_19;
        if ( !v13 )
          v13 = v12;
        goto LABEL_29;
      }
      if ( !a3 )
        goto LABEL_19;
LABEL_33:
      while ( *(_BYTE *)(a3 + 140) == 12 )
        a3 = *(_QWORD *)(a3 + 160);
      if ( a5 == 0 && dword_4F077C4 != 2 && v22 != 32 )
        goto LABEL_19;
      if ( v15[1] == unk_4F07290 )
        goto LABEL_19;
      while ( v19 == 12 )
      {
        v18 = *(_QWORD *)(v18 + 160);
        v19 = *(_BYTE *)(v18 + 140);
      }
      if ( v19 )
      {
        v31 = v13;
        v33 = v10;
        v40 = v9;
        v25 = sub_8DE890(a3, v20, 0, 0);
        v9 = v40;
        v10 = v33;
        v13 = v31;
        if ( v25 )
        {
          v26 = *(__int128 **)(v15[1] + 216LL);
          if ( v26 == a4 || (v27 = sub_739400(v26, a4), v9 = v40, v10 = v33, v13 = v31, v27) )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(a3 + 168) + 20LL) & 2) == 0
              && (*(_BYTE *)(*(_QWORD *)(v20 + 168) + 20LL) & 2) == 0
              || (v32 = v13, v34 = v10, v41 = v9, v28 = sub_8D73A0(a3, v20), v9 = v41, v10 = v34, v13 = v32, v28) )
            {
LABEL_19:
              v7 = a1;
              v6 = v37;
              v23 = v12;
              goto LABEL_20;
            }
          }
        }
      }
LABEL_5:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        goto LABEL_30;
    }
    if ( v22 != 48 )
    {
      if ( !v21 && v16 )
        goto LABEL_19;
      goto LABEL_16;
    }
    if ( v10 == v21 && a3 != 0 && !v16 )
    {
      if ( v19 == 12 )
      {
        v29 = *v15;
        do
        {
          v29 = *(_QWORD *)(v29 + 160);
          v30 = *(_BYTE *)(v29 + 140);
        }
        while ( v30 == 12 );
      }
      else
      {
        v30 = *(_BYTE *)(*v15 + 140LL);
      }
      if ( v30 )
      {
        v39 = v9;
        if ( v14 != 14 )
          goto LABEL_33;
      }
    }
    if ( !v13 )
      v13 = v12;
LABEL_29:
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 );
LABEL_30:
  v23 = v13;
  v7 = a1;
  v6 = v37;
  if ( v39 )
    goto LABEL_31;
LABEL_20:
  v6->m128i_i64[1] = v7->m128i_i64[1];
  return v23;
}
