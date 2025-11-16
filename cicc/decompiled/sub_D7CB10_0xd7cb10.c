// Function: sub_D7CB10
// Address: 0xd7cb10
//
__int64 __fastcall sub_D7CB10(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int8 *v16; // rbx
  unsigned __int8 *v17; // r12
  __m128i *v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rax
  char *v21; // rsi
  char *v22; // rax
  char *v23; // rax
  char *v24; // rax
  __int64 result; // rax
  unsigned int v26; // esi
  __int64 v27; // r9
  unsigned int v28; // r8d
  __int64 v29; // r11
  __int64 v30; // rdx
  int v31; // r12d
  __int64 *v32; // rdi
  __int64 v33; // rcx
  unsigned int v34; // r8d
  int v35; // edx
  __m128i *v36; // rsi
  __m128i *v37; // rax
  __int64 v38; // rdx
  __m128i *v39; // rsi
  int v40; // eax
  __m128i *v41; // [rsp+8h] [rbp-A8h]
  __int64 *v44; // [rsp+28h] [rbp-88h] BYREF
  char *v45; // [rsp+30h] [rbp-80h] BYREF
  char *v46; // [rsp+38h] [rbp-78h]
  char *v47; // [rsp+40h] [rbp-70h]
  __m128i v48; // [rsp+50h] [rbp-60h] BYREF
  char *v49[2]; // [rsp+60h] [rbp-50h] BYREF
  char *v50; // [rsp+70h] [rbp-40h]

  v7 = *a2;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v9 = sub_BD2BC0((__int64)a2);
    v11 = v9 + v10;
    v12 = 0;
    if ( (a2[7] & 0x80u) != 0 )
      v12 = sub_BD2BC0((__int64)a2);
    if ( (unsigned int)((v11 - v12) >> 4) )
    {
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v14 = sub_BD2BC0((__int64)a2);
      v8 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
    }
  }
  v16 = &a2[v8];
  v17 = &a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  if ( v16 == v17 )
  {
LABEL_19:
    v48.m128i_i64[0] = a3;
    v48.m128i_i64[1] = a1;
    v22 = v45;
    v45 = 0;
    v49[0] = v22;
    v23 = v46;
    v46 = 0;
    v49[1] = v23;
    v24 = v47;
    v47 = 0;
    v50 = v24;
    result = sub_D79E80((_DWORD *)a5, v48.m128i_i64, &v44);
    if ( !(_BYTE)result )
    {
      v37 = (__m128i *)sub_D7C840(a5, v48.m128i_i64, v44);
      *v37 = _mm_loadu_si128(&v48);
      sub_D76A50((__int64)v37[1].m128i_i64, v49);
      v39 = *(__m128i **)(a5 + 40);
      if ( v39 == *(__m128i **)(a5 + 48) )
      {
        result = (__int64)sub_D78F50(a5 + 32, v39, &v48);
      }
      else
      {
        if ( v39 )
        {
          *v39 = _mm_loadu_si128(&v48);
          sub_D78650((__m128i *)v39[1].m128i_i64, (const void **)v49, v38);
          v39 = *(__m128i **)(a5 + 40);
        }
        result = a5;
        *(_QWORD *)(a5 + 40) = (char *)v39 + 40;
      }
    }
    if ( v49[0] )
      result = j_j___libc_free_0(v49[0], v50 - v49[0]);
    goto LABEL_22;
  }
  v18 = &v48;
  while ( 1 )
  {
    v19 = *(_BYTE **)v17;
    if ( **(_BYTE **)v17 != 17 || *((_DWORD *)v19 + 8) > 0x40u )
      break;
    v20 = *((_QWORD *)v19 + 3);
    v21 = v46;
    v48.m128i_i64[0] = v20;
    if ( v46 == v47 )
    {
      v41 = v18;
      sub_A235E0((__int64)&v45, v46, v18);
      v18 = v41;
    }
    else
    {
      if ( v46 )
      {
        *(_QWORD *)v46 = v20;
        v21 = v46;
      }
      v46 = v21 + 8;
    }
    v17 += 32;
    if ( v16 == v17 )
      goto LABEL_19;
  }
  v26 = *(_DWORD *)(a4 + 24);
  v48.m128i_i64[0] = a3;
  v48.m128i_i64[1] = a1;
  if ( !v26 )
  {
    ++*(_QWORD *)a4;
    v44 = 0;
    goto LABEL_43;
  }
  v27 = *(_QWORD *)(a4 + 8);
  v28 = a3 & (v26 - 1);
  result = v27 + 16LL * v28;
  v29 = *(_QWORD *)(result + 8);
  v30 = *(_QWORD *)result;
  if ( a1 == v29 && a3 == v30 )
    goto LABEL_22;
  v31 = 1;
  v32 = 0;
  while ( 1 )
  {
    if ( v30 )
      goto LABEL_32;
    if ( v29 == -1 )
      break;
    if ( v29 == -2 && !v32 )
      v32 = (__int64 *)result;
LABEL_32:
    v28 = (v26 - 1) & (v31 + v28);
    result = v27 + 16LL * v28;
    v30 = *(_QWORD *)result;
    v29 = *(_QWORD *)(result + 8);
    if ( a1 == v29 && a3 == v30 )
      goto LABEL_22;
    ++v31;
  }
  if ( !v32 )
    v32 = (__int64 *)result;
  v40 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v35 = v40 + 1;
  v44 = v32;
  if ( 4 * (v40 + 1) >= 3 * v26 )
  {
LABEL_43:
    v26 *= 2;
    goto LABEL_44;
  }
  result = v26 - *(_DWORD *)(a4 + 20) - v35;
  if ( (unsigned int)result <= v26 >> 3 )
  {
LABEL_44:
    sub_D7B870(a4, v26);
    sub_D79DB0(a4, v48.m128i_i64, &v44, v33, v34);
    result = *(unsigned int *)(a4 + 16);
    v32 = v44;
    v35 = result + 1;
  }
  *(_DWORD *)(a4 + 16) = v35;
  if ( v32[1] != -1 || *v32 )
    --*(_DWORD *)(a4 + 20);
  *(__m128i *)v32 = _mm_loadu_si128(&v48);
  v36 = *(__m128i **)(a4 + 40);
  if ( v36 == *(__m128i **)(a4 + 48) )
  {
    result = (__int64)sub_D78DC0(a4 + 32, v36, &v48);
  }
  else
  {
    if ( v36 )
    {
      *v36 = _mm_loadu_si128(&v48);
      v36 = *(__m128i **)(a4 + 40);
    }
    *(_QWORD *)(a4 + 40) = v36 + 1;
  }
LABEL_22:
  if ( v45 )
    return j_j___libc_free_0(v45, v47 - v45);
  return result;
}
