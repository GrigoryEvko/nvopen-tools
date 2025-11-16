// Function: sub_9CA790
// Address: 0x9ca790
//
__int64 __fastcall sub_9CA790(_QWORD *a1, const void *a2, size_t a3)
{
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __m128i v13; // xmm1
  __int64 v14; // rdx
  __int64 v15; // rbx
  _QWORD *v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // [rsp+8h] [rbp-178h]
  __m128i v22[13]; // [rsp+10h] [rbp-170h] BYREF
  __int64 v23[3]; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v24; // [rsp+F8h] [rbp-88h]
  __int128 v25; // [rsp+108h] [rbp-78h]
  __int64 v26; // [rsp+118h] [rbp-68h]
  int v27; // [rsp+128h] [rbp-58h] BYREF
  _QWORD *v28; // [rsp+130h] [rbp-50h]
  int *v29; // [rsp+138h] [rbp-48h]
  int *v30; // [rsp+140h] [rbp-40h]
  __int64 v31; // [rsp+148h] [rbp-38h]

  v6 = a1 + 27;
  v7 = sub_B2F650(a2, a3);
  v8 = a1[28];
  v9 = v7;
  v21 = a1 + 26;
  if ( !v8 )
    goto LABEL_7;
  while ( 1 )
  {
    while ( v9 > *(_QWORD *)(v8 + 32) )
    {
      v8 = *(_QWORD *)(v8 + 24);
      if ( !v8 )
        goto LABEL_7;
    }
    v10 = *(_QWORD **)(v8 + 16);
    if ( v9 >= *(_QWORD *)(v8 + 32) )
      break;
    v6 = (_QWORD *)v8;
    v8 = *(_QWORD *)(v8 + 16);
    if ( !v10 )
      goto LABEL_7;
  }
  v17 = *(_QWORD **)(v8 + 24);
  if ( v17 )
  {
    do
    {
      while ( 1 )
      {
        v18 = v17[2];
        v19 = v17[3];
        if ( v9 < v17[4] )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v19 )
          goto LABEL_13;
      }
      v6 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( v18 );
  }
LABEL_13:
  while ( v10 )
  {
    while ( 1 )
    {
      v20 = v10[3];
      if ( v9 <= v10[4] )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v20 )
        goto LABEL_16;
    }
    v8 = (__int64)v10;
    v10 = (_QWORD *)v10[2];
  }
LABEL_16:
  if ( v6 != (_QWORD *)v8 )
  {
    while ( *(_QWORD *)(v8 + 48) != a3 || a3 && memcmp(*(const void **)(v8 + 40), a2, a3) )
    {
      v8 = sub_220EEE0(v8);
      if ( (_QWORD *)v8 == v6 )
        goto LABEL_7;
    }
    return v8 + 56;
  }
  else
  {
LABEL_7:
    v11 = sub_B2F650(a2, a3);
    v12 = sub_C94910(a1 + 21, a2, a3);
    v22[0] = 0;
    v22[0].m128i_i32[0] = 5;
    v13 = _mm_loadu_si128(v22);
    v23[1] = v12;
    v23[2] = v14;
    v24 = v13;
    v25 = 0;
    v23[0] = v11;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = &v27;
    v30 = &v27;
    v31 = 0;
    v15 = sub_9CA630(v21, v23);
    sub_9C4830(v28);
    sub_9C4830(0);
    sub_9C4830(0);
    return v15 + 56;
  }
}
