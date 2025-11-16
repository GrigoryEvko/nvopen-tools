// Function: sub_15DC8A0
// Address: 0x15dc8a0
//
__int64 __fastcall sub_15DC8A0(const __m128i *a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 *v6; // r12
  __int64 v8; // r14
  __int64 i; // r15
  __int64 result; // rax
  __int64 v11; // rax
  char v12; // al
  __int64 *v13; // rcx
  int v14; // r8d
  __int64 *v15; // rcx
  unsigned __int64 v16; // rdx
  bool v17; // zf
  char v18; // al
  __int64 v19; // r9
  int v20; // esi
  unsigned int v21; // r10d
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  int v24; // eax
  __int64 *v25; // rdi
  unsigned int j; // eax
  __int64 v27; // r10
  __int64 *v28; // r11
  unsigned int v29; // esi
  unsigned int v30; // eax
  int v31; // eax
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 *v34; // rcx
  __int64 v35; // r8
  unsigned int v36; // eax
  int v37; // edx
  unsigned int v38; // r9d
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // [rsp+0h] [rbp-80h]
  int v42; // [rsp+Ch] [rbp-74h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  char v44; // [rsp+10h] [rbp-70h]
  int v45; // [rsp+10h] [rbp-70h]
  __int64 *v47; // [rsp+28h] [rbp-58h] BYREF
  __int64 v48; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v49; // [rsp+38h] [rbp-48h]
  __int64 *v50; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v51; // [rsp+48h] [rbp-38h]

  v4 = (char *)a2 - (char *)a1;
  v6 = a2;
  v43 = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 > 16 )
  {
    v8 = v4 >> 4;
    for ( i = ((v4 >> 4) - 2) / 2; ; --i )
    {
      sub_15DC190((__int64)a1, i, v8, (__int64 *)a1[i].m128i_i64[0], a1[i].m128i_i64[1], a4);
      if ( !i )
        break;
    }
  }
  result = v43 >> 4;
  v41 = v43 >> 4;
  if ( (unsigned __int64)a2 >= a3 )
    return result;
  do
  {
    v11 = v6[1];
    v48 = *v6;
    v49 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = sub_15D0A10(a4, &v48, &v50);
    v13 = v50;
    if ( !v12 )
    {
      v30 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v31 = (v30 >> 1) + 1;
      if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
      {
        v32 = 4;
        if ( (unsigned int)(4 * v31) < 0xC )
        {
LABEL_22:
          if ( v32 - (v31 + *(_DWORD *)(a4 + 12)) > v32 >> 3 )
            goto LABEL_23;
          goto LABEL_41;
        }
      }
      else
      {
        v32 = *(_DWORD *)(a4 + 24);
        if ( 4 * v31 < 3 * v32 )
          goto LABEL_22;
      }
      v32 *= 2;
LABEL_41:
      sub_15D0B40(a4, v32);
      sub_15D0A10(a4, &v48, &v50);
      v13 = v50;
      v31 = (*(_DWORD *)(a4 + 8) >> 1) + 1;
LABEL_23:
      *(_DWORD *)(a4 + 8) = *(_DWORD *)(a4 + 8) & 1 | (2 * v31);
      if ( *v13 != -8 || v13[1] != -8 )
        --*(_DWORD *)(a4 + 12);
      v33 = v48;
      *((_DWORD *)v13 + 4) = 0;
      v14 = 0;
      *v13 = v33;
      v13[1] = v49;
      goto LABEL_8;
    }
    v14 = *((_DWORD *)v50 + 4);
LABEL_8:
    v15 = (__int64 *)a1->m128i_i64[0];
    v16 = a1->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
    v17 = (*(_BYTE *)(a4 + 8) & 1) == 0;
    v18 = *(_BYTE *)(a4 + 8) & 1;
    v50 = (__int64 *)a1->m128i_i64[0];
    v51 = v16;
    v44 = v18;
    if ( v17 )
    {
      v29 = *(_DWORD *)(a4 + 24);
      v19 = *(_QWORD *)(a4 + 16);
      if ( !v29 )
      {
        v36 = *(_DWORD *)(a4 + 8);
        ++*(_QWORD *)a4;
        v25 = 0;
        v37 = (v36 >> 1) + 1;
        goto LABEL_33;
      }
      v20 = v29 - 1;
    }
    else
    {
      v19 = a4 + 16;
      v20 = 3;
    }
    v42 = 1;
    v21 = (unsigned int)v16 >> 9;
    v22 = (((v21 ^ ((unsigned int)v16 >> 4)
           | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v21 ^ ((unsigned int)v16 >> 4)) << 32)) >> 22)
        ^ ((v21 ^ ((unsigned int)v16 >> 4)
          | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v21 ^ ((unsigned int)v16 >> 4)) << 32));
    v23 = 9 * (((v22 - 1 - (v22 << 13)) >> 8) ^ (v22 - 1 - (v22 << 13)));
    v24 = (((v23 ^ (v23 >> 15)) - 1 - ((v23 ^ (v23 >> 15)) << 27)) >> 31)
        ^ ((v23 ^ (v23 >> 15)) - 1 - (((unsigned int)v23 ^ (unsigned int)(v23 >> 15)) << 27));
    v25 = 0;
    for ( j = v20 & v24; ; j = v20 & v40 )
    {
      v27 = v19 + 24LL * j;
      v28 = *(__int64 **)v27;
      if ( v15 == *(__int64 **)v27 && v16 == *(_QWORD *)(v27 + 8) )
      {
        result = *(unsigned int *)(v27 + 16);
        goto LABEL_28;
      }
      if ( v28 == (__int64 *)-8LL )
        break;
      if ( v28 == (__int64 *)-16LL && *(_QWORD *)(v27 + 8) == -16 && !v25 )
        v25 = (__int64 *)(v19 + 24LL * j);
LABEL_50:
      v40 = v42 + j;
      ++v42;
    }
    if ( *(_QWORD *)(v27 + 8) != -8 )
      goto LABEL_50;
    v36 = *(_DWORD *)(a4 + 8);
    v38 = 12;
    v29 = 4;
    if ( !v25 )
      v25 = (__int64 *)v27;
    ++*(_QWORD *)a4;
    v37 = (v36 >> 1) + 1;
    if ( !v44 )
    {
      v29 = *(_DWORD *)(a4 + 24);
LABEL_33:
      v38 = 3 * v29;
    }
    if ( v38 <= 4 * v37 )
    {
      v45 = v14;
      v29 *= 2;
    }
    else
    {
      if ( v29 - *(_DWORD *)(a4 + 12) - v37 > v29 >> 3 )
        goto LABEL_36;
      v45 = v14;
    }
    sub_15D0B40(a4, v29);
    sub_15D0A10(a4, (__int64 *)&v50, &v47);
    v25 = v47;
    v15 = v50;
    v36 = *(_DWORD *)(a4 + 8);
    v14 = v45;
LABEL_36:
    *(_DWORD *)(a4 + 8) = (2 * (v36 >> 1) + 2) | v36 & 1;
    if ( *v25 != -8 || v25[1] != -8 )
      --*(_DWORD *)(a4 + 12);
    *v25 = (__int64)v15;
    v39 = v51;
    *((_DWORD *)v25 + 4) = 0;
    v25[1] = v39;
    result = 0;
LABEL_28:
    if ( v14 > (int)result )
    {
      v34 = (__int64 *)*v6;
      v35 = v6[1];
      *(__m128i *)v6 = _mm_loadu_si128(a1);
      result = sub_15DC190((__int64)a1, 0, v41, v34, v35, a4);
    }
    v6 += 2;
  }
  while ( a3 > (unsigned __int64)v6 );
  return result;
}
