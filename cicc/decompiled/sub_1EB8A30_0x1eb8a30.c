// Function: sub_1EB8A30
// Address: 0x1eb8a30
//
__int64 __fastcall sub_1EB8A30(__int64 a1, __int64 a2, __int16 a3, unsigned int a4, signed int a5, __int64 a6)
{
  __int64 v10; // rbx
  int v11; // esi
  __int64 v12; // r8
  unsigned int v13; // ecx
  __int64 v14; // r10
  _BYTE *v15; // r10
  unsigned int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int16 v22; // si
  __int64 v24; // rax
  __m128i *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  char v30; // al
  __int64 v31; // rax
  __int64 i; // rax
  __int64 v33; // rcx
  __int16 v34; // ax
  __int64 v35; // [rsp+0h] [rbp-60h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+8h] [rbp-58h]
  __m128i v39; // [rsp+10h] [rbp-50h] BYREF
  __int64 v40; // [rsp+20h] [rbp-40h]

  v10 = a4;
  v11 = a4 & 0x7FFFFFFF;
  v12 = a4 & 0x7FFFFFFF;
  v13 = *(_DWORD *)(a1 + 400);
  v39.m128i_i64[1] = (unsigned int)v10;
  v14 = *(_QWORD *)(a1 + 600);
  v39.m128i_i64[0] = 0;
  v15 = (_BYTE *)(v12 + v14);
  LOBYTE(v40) = 0;
  v16 = (unsigned __int8)*v15;
  if ( v16 >= v13 )
    goto LABEL_16;
  v17 = *(_QWORD *)(a1 + 392);
  while ( 1 )
  {
    a6 = v17 + 24LL * v16;
    v18 = *(_DWORD *)(a6 + 8) & 0x7FFFFFFF;
    if ( v11 == (_DWORD)v18 )
      break;
    v16 += 256;
    if ( v13 <= v16 )
      goto LABEL_16;
  }
  if ( a6 == v17 + 24LL * v13 )
  {
LABEL_16:
    *v15 = v13;
    v24 = *(unsigned int *)(a1 + 400);
    if ( (unsigned int)v24 >= *(_DWORD *)(a1 + 404) )
    {
      v38 = v12;
      sub_16CD150(a1 + 392, (const void *)(a1 + 408), 0, 24, v12, a6);
      v24 = *(unsigned int *)(a1 + 400);
      v12 = v38;
    }
    v25 = (__m128i *)(*(_QWORD *)(a1 + 392) + 24 * v24);
    v26 = v40;
    *v25 = _mm_loadu_si128(&v39);
    v25[1].m128i_i64[0] = v26;
    v27 = (unsigned int)(*(_DWORD *)(a1 + 400) + 1);
    *(_DWORD *)(a1 + 400) = v27;
    v28 = *(_QWORD *)(a1 + 392) + 24 * v27 - 24;
    if ( a5 <= 0 )
    {
      v35 = v12;
      v37 = *(_QWORD *)(a1 + 392) + 24 * v27 - 24;
      v30 = sub_1E69E00(*(_QWORD *)(a1 + 240), v10);
      v28 = v37;
      if ( v30 )
      {
        v31 = *(_QWORD *)(a1 + 240);
        for ( i = (int)v10 < 0
                ? *(_QWORD *)(*(_QWORD *)(v31 + 24) + 16 * v35 + 8)
                : *(_QWORD *)(*(_QWORD *)(v31 + 272) + 8 * v10);
              i && ((*(_BYTE *)(i + 3) & 0x10) != 0 || (*(_BYTE *)(i + 4) & 8) != 0);
              i = *(_QWORD *)(i + 32) )
        {
          ;
        }
        v33 = *(_QWORD *)(i + 16);
        v34 = **(_WORD **)(v33 + 16);
        if ( v34 == 10 || v34 == 15 )
          a5 = *(_DWORD *)(*(_QWORD *)(v33 + 32) + 8LL);
      }
    }
    v29 = sub_1EB8380(a1, a2, v28, a5);
    v22 = *(_WORD *)(v29 + 12);
    a6 = v29;
  }
  else
  {
    v19 = *(_QWORD *)a6;
    v20 = *(unsigned __int16 *)(a6 + 12);
    if ( !*(_QWORD *)a6 )
    {
LABEL_9:
      v22 = *(_WORD *)(a6 + 12);
      goto LABEL_10;
    }
    v18 = 40LL * *(unsigned __int16 *)(a6 + 14);
    if ( v19 == a2 )
    {
      v21 = v18 + *(_QWORD *)(a2 + 32);
      if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
        goto LABEL_9;
    }
    else
    {
      v18 += *(_QWORD *)(v19 + 32);
      v21 = v18;
      if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
        goto LABEL_9;
    }
    if ( *(_BYTE *)v21 || (v22 = *(_WORD *)(a6 + 12), (*(_WORD *)(v21 + 2) & 0xFF0) == 0) )
    {
      v22 = *(_WORD *)(a6 + 12);
      if ( *(_DWORD *)(v21 + 8) == (unsigned __int16)v20 )
      {
        *(_BYTE *)(v21 + 3) |= 0x40u;
        v22 = *(_WORD *)(a6 + 12);
      }
    }
  }
LABEL_10:
  *(_QWORD *)a6 = a2;
  *(_WORD *)(a6 + 14) = a3;
  *(_BYTE *)(a6 + 16) = 1;
  v36 = a6;
  sub_1EB6840(a1, v22, v18, v20, v12, a6);
  return v36;
}
