// Function: sub_2D2F750
// Address: 0x2d2f750
//
void __fastcall sub_2D2F750(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  int v11; // eax
  int v12; // ecx
  __int64 v13; // rbx
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rbx
  bool v18; // zf
  __m128i *v19; // rax
  unsigned int v20; // esi
  int v21; // eax
  int v22; // eax
  __int64 v23; // rdx
  unsigned __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  bool v29; // al
  unsigned __int64 v30; // [rsp+8h] [rbp-128h]
  char v31; // [rsp+17h] [rbp-119h]
  int v32; // [rsp+20h] [rbp-110h]
  int v33; // [rsp+24h] [rbp-10Ch]
  unsigned __int64 v34; // [rsp+38h] [rbp-F8h]
  unsigned int v35; // [rsp+40h] [rbp-F0h]
  __int64 v36; // [rsp+40h] [rbp-F0h]
  int v37; // [rsp+5Ch] [rbp-D4h] BYREF
  __int64 v38; // [rsp+60h] [rbp-D0h] BYREF
  __m128i *v39; // [rsp+68h] [rbp-C8h] BYREF
  __m128i v40; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v41; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+90h] [rbp-A0h]
  __int64 v43[3]; // [rsp+A0h] [rbp-90h] BYREF
  char v44; // [rsp+B8h] [rbp-78h]
  __int64 v45; // [rsp+C0h] [rbp-70h]
  __int64 v46[2]; // [rsp+D0h] [rbp-60h] BYREF
  char v47; // [rsp+E0h] [rbp-50h] BYREF

  v2 = *a1;
  v46[0] = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v30 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v3 = sub_2D2B870((_QWORD *)(v2 + 72), v46);
  if ( !v3 )
    return;
  v31 = 0;
  v46[1] = 0x100000000LL;
  v46[0] = (__int64)&v47;
  v4 = *(_QWORD *)(v3 + 16);
  v34 = v4 + 32LL * *(unsigned int *)(v3 + 24);
  if ( v4 == v34 )
    goto LABEL_21;
  do
  {
    v5 = sub_B10D40(v4 + 16);
    v6 = a1[1];
    v7 = v5;
    v8 = *(_QWORD *)(*(_QWORD *)(*a1 + 48) + 40LL * (unsigned int)(*(_DWORD *)v4 - 1));
    v41.m128i_i8[8] = 0;
    v42 = v7;
    v9 = *(unsigned int *)(v6 + 24);
    v40.m128i_i64[0] = v8;
    v10 = *(_QWORD *)(v6 + 8);
    if ( (_DWORD)v9 )
    {
      v39 = (__m128i *)v7;
      v43[0] = 0;
      v44 = 0;
      v45 = 0;
      v37 = 0;
      v38 = v8;
      v11 = sub_F11290(&v38, &v37, (__int64 *)&v39);
      v12 = v9 - 1;
      v13 = v40.m128i_i64[0];
      v33 = 1;
      v32 = v12;
      v35 = v12 & v11;
      while ( 1 )
      {
        v14 = v10 + 56LL * v35;
        if ( *(_QWORD *)v14 == v13 && v41.m128i_i8[8] == *(_BYTE *)(v14 + 24) )
        {
          if ( !v41.m128i_i8[8]
            || (v29 = sub_2D27C10(&v40.m128i_i64[1], (_QWORD *)(v14 + 8)), v14 = v10 + 56LL * v35, v29) )
          {
            if ( v42 == *(_QWORD *)(v14 + 32) )
              break;
          }
        }
        if ( sub_F34140(v14, (__int64)v43) )
        {
          v10 = *(_QWORD *)(v6 + 8);
          v9 = *(unsigned int *)(v6 + 24);
          v6 = a1[1];
          v15 = *(_QWORD *)(v6 + 8);
          v16 = *(unsigned int *)(v6 + 24);
          goto LABEL_8;
        }
        v35 = v32 & (v33 + v35);
        ++v33;
      }
      v6 = a1[1];
      v15 = *(_QWORD *)(v6 + 8);
      v16 = *(unsigned int *)(v6 + 24);
    }
    else
    {
      v16 = 0;
      v15 = *(_QWORD *)(v6 + 8);
LABEL_8:
      v14 = v10 + 56 * v9;
    }
    v17 = *(_QWORD *)(v4 + 8);
    if ( v14 == v15 + 56 * v16 || *(_QWORD *)(v14 + 40) != *(_QWORD *)(v4 + 24) || *(_QWORD *)(v14 + 48) != v17 )
    {
      v36 = *(_QWORD *)(v4 + 24);
      v18 = (unsigned __int8)sub_2D29210(v6, (__int64)&v40, (__int64 *)&v39) == 0;
      v19 = v39;
      if ( !v18 )
      {
LABEL_17:
        v19[3].m128i_i64[0] = v17;
        v19[2].m128i_i64[1] = v36;
        sub_2D29B40((unsigned int *)v46, v4);
        goto LABEL_18;
      }
      v43[0] = (__int64)v39;
      v20 = *(_DWORD *)(v6 + 24);
      v21 = *(_DWORD *)(v6 + 16);
      ++*(_QWORD *)v6;
      v22 = v21 + 1;
      if ( 4 * v22 >= 3 * v20 )
      {
        v20 *= 2;
      }
      else if ( v20 - *(_DWORD *)(v6 + 20) - v22 > v20 >> 3 )
      {
        goto LABEL_14;
      }
      sub_2D2F0E0(v6, v20);
      sub_2D29210(v6, (__int64)&v40, v43);
      v22 = *(_DWORD *)(v6 + 16) + 1;
LABEL_14:
      *(_DWORD *)(v6 + 16) = v22;
      v19 = (__m128i *)v43[0];
      if ( *(_QWORD *)v43[0] || *(_BYTE *)(v43[0] + 24) || *(_QWORD *)(v43[0] + 32) )
        --*(_DWORD *)(v6 + 20);
      *v19 = _mm_loadu_si128(&v40);
      v19[1] = _mm_loadu_si128(&v41);
      v23 = v42;
      v19[2].m128i_i64[1] = 0;
      v19[2].m128i_i64[0] = v23;
      v19[3].m128i_i64[0] = 0;
      goto LABEL_17;
    }
    v31 = 1;
LABEL_18:
    v4 += 32LL;
  }
  while ( v34 != v4 );
  if ( v31 )
  {
    v43[0] = v30;
    v24 = sub_2D2B8B0((unsigned __int64 *)(*a1 + 72), v43);
    sub_2D29780((unsigned int *)v24, (__int64)v46, v25, v26, v27, v28);
    *(_BYTE *)a1[2] = 1;
  }
LABEL_21:
  sub_2D288B0((__int64)v46);
}
