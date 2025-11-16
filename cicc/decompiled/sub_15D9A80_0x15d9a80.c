// Function: sub_15D9A80
// Address: 0x15d9a80
//
__int64 *__fastcall sub_15D9A80(const __m128i *a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned int v4; // esi
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  const __m128i *v8; // rbx
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // rdi
  int v12; // esi
  int v13; // r12d
  __int64 *v14; // rdx
  unsigned int i; // r11d
  __int64 v16; // r9
  __int64 v17; // rcx
  unsigned int v18; // r11d
  unsigned int v19; // esi
  int v20; // r12d
  __int64 v21; // rax
  char v22; // al
  __int64 *v23; // rsi
  unsigned int v24; // eax
  int v25; // eax
  unsigned int v26; // r9d
  unsigned __int64 v27; // rax
  unsigned int v29; // edi
  int v30; // r9d
  unsigned int v31; // r10d
  unsigned int v32; // eax
  __int64 v33; // r9
  unsigned __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // [rsp+8h] [rbp-88h]
  int v37; // [rsp+14h] [rbp-7Ch]
  __int64 *v38; // [rsp+18h] [rbp-78h]
  __int64 *v39; // [rsp+38h] [rbp-58h] BYREF
  __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 *v42; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v43; // [rsp+58h] [rbp-38h]

  v3 = a1->m128i_i64[0];
  v36 = a1->m128i_i64[1];
  v4 = (unsigned int)v36 >> 9;
  v5 = (((v4 ^ ((unsigned int)v36 >> 4) | ((unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v4 ^ ((unsigned int)v36 >> 4)) << 32)) >> 22)
     ^ ((v4 ^ ((unsigned int)v36 >> 4) | ((unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32))
      - 1
      - ((unsigned __int64)(v4 ^ ((unsigned int)v36 >> 4)) << 32));
  v6 = ((9 * (((v5 - 1 - (v5 << 13)) >> 8) ^ (v5 - 1 - (v5 << 13)))) >> 15)
     ^ (9 * (((v5 - 1 - (v5 << 13)) >> 8) ^ (v5 - 1 - (v5 << 13))));
  v37 = ((v6 - 1 - (v6 << 27)) >> 31) ^ (v6 - 1 - ((_DWORD)v6 << 27));
  v7 = a2;
  v8 = a1;
  v9 = v7;
  while ( 2 )
  {
    v38 = (__int64 *)v8;
    v40 = v3;
    v41 = v36 & 0xFFFFFFFFFFFFFFF8LL;
    v10 = *(_BYTE *)(v9 + 8) & 1;
    if ( v10 )
    {
      v11 = v9 + 16;
      v12 = 3;
    }
    else
    {
      v19 = *(_DWORD *)(v9 + 24);
      v11 = *(_QWORD *)(v9 + 16);
      if ( !v19 )
      {
        v29 = *(_DWORD *)(v9 + 8);
        ++*(_QWORD *)v9;
        v14 = 0;
        v30 = (v29 >> 1) + 1;
        goto LABEL_27;
      }
      v12 = v19 - 1;
    }
    v13 = 1;
    v14 = 0;
    for ( i = v12 & v37; ; i = v12 & v18 )
    {
      v16 = v11 + 24LL * i;
      v17 = *(_QWORD *)v16;
      if ( v3 == *(_QWORD *)v16 && *(_QWORD *)(v16 + 8) == (v36 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v20 = *(_DWORD *)(v16 + 16);
        goto LABEL_16;
      }
      if ( v17 == -8 )
        break;
      if ( v17 == -16 && *(_QWORD *)(v16 + 8) == -16 && !v14 )
        v14 = (__int64 *)(v11 + 24LL * i);
LABEL_11:
      v18 = v13 + i;
      ++v13;
    }
    if ( *(_QWORD *)(v16 + 8) != -8 )
      goto LABEL_11;
    v29 = *(_DWORD *)(v9 + 8);
    v31 = 12;
    v19 = 4;
    if ( !v14 )
      v14 = (__int64 *)v16;
    ++*(_QWORD *)v9;
    v30 = (v29 >> 1) + 1;
    if ( !v10 )
    {
      v19 = *(_DWORD *)(v9 + 24);
LABEL_27:
      v31 = 3 * v19;
    }
    if ( 4 * v30 >= v31 )
    {
      v19 *= 2;
    }
    else
    {
      v32 = v19 - *(_DWORD *)(v9 + 12) - v30;
      v33 = v3;
      if ( v32 > v19 >> 3 )
        goto LABEL_30;
    }
    sub_15D0B40(v9, v19);
    sub_15D0A10(v9, &v40, &v42);
    v14 = v42;
    v33 = v40;
    v29 = *(_DWORD *)(v9 + 8);
LABEL_30:
    *(_DWORD *)(v9 + 8) = (2 * (v29 >> 1) + 2) | v29 & 1;
    if ( *v14 != -8 || v14[1] != -8 )
      --*(_DWORD *)(v9 + 12);
    *v14 = v33;
    v34 = v41;
    v20 = 0;
    *((_DWORD *)v14 + 4) = 0;
    v14[1] = v34;
LABEL_16:
    v21 = v8[-1].m128i_i64[1];
    v42 = (__int64 *)v8[-1].m128i_i64[0];
    v43 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    v22 = sub_15D0A10(v9, (__int64 *)&v42, &v39);
    v23 = v39;
    if ( v22 )
    {
      --v8;
      if ( v20 > *((_DWORD *)v39 + 4) )
        goto LABEL_18;
      break;
    }
    v24 = *(_DWORD *)(v9 + 8);
    ++*(_QWORD *)v9;
    v25 = (v24 >> 1) + 1;
    if ( (*(_BYTE *)(v9 + 8) & 1) != 0 )
    {
      v26 = 4;
      if ( (unsigned int)(4 * v25) >= 0xC )
        goto LABEL_34;
LABEL_21:
      if ( v26 - (v25 + *(_DWORD *)(v9 + 12)) <= v26 >> 3 )
      {
        v35 = v26;
        goto LABEL_35;
      }
    }
    else
    {
      v26 = *(_DWORD *)(v9 + 24);
      if ( 3 * v26 > 4 * v25 )
        goto LABEL_21;
LABEL_34:
      v35 = 2 * v26;
LABEL_35:
      sub_15D0B40(v9, v35);
      sub_15D0A10(v9, (__int64 *)&v42, &v39);
      v23 = v39;
      v25 = (*(_DWORD *)(v9 + 8) >> 1) + 1;
    }
    *(_DWORD *)(v9 + 8) = *(_DWORD *)(v9 + 8) & 1 | (2 * v25);
    if ( *v23 != -8 || v23[1] != -8 )
      --*(_DWORD *)(v9 + 12);
    --v8;
    *v23 = (__int64)v42;
    v27 = v43;
    *((_DWORD *)v23 + 4) = 0;
    v23[1] = v27;
    if ( v20 > 0 )
    {
LABEL_18:
      v8[1] = _mm_loadu_si128(v8);
      continue;
    }
    break;
  }
  *v38 = v3;
  v38[1] = v36;
  return v38;
}
