// Function: sub_2257D90
// Address: 0x2257d90
//
__int64 __fastcall sub_2257D90(__int64 a1, __int64 *a2, unsigned int a3, char a4)
{
  __int64 *v5; // r13
  unsigned int v7; // ebx
  char *v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  bool v12; // cf
  unsigned __int64 v13; // rax
  char *v14; // rax
  const void *v15; // r15
  signed __int64 v16; // r8
  __int64 *v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // r9
  int v21; // esi
  int v22; // edx
  __int64 v23; // r11
  __int64 v24; // rcx
  char v25; // cl
  __int64 *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 result; // rax
  int v33; // edi
  int v34; // esi
  __int64 *v35; // r8
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rax
  char *v43; // [rsp+10h] [rbp-50h]
  unsigned __int64 v44; // [rsp+18h] [rbp-48h]
  unsigned int v45; // [rsp+28h] [rbp-38h]

  v5 = a2;
  v7 = a3;
  v8 = *(char **)(a1 + 16);
  v9 = *(unsigned int *)(a1 + 24);
  if ( v8 != *(char **)(a1 + 32) )
  {
    v33 = *(_DWORD *)(a1 + 24);
    v34 = (v9 + 1) & 0x3F;
    v35 = (__int64 *)&v8[8 * ((v9 + 1) >> 6)];
    v36 = v9 + 8 * (v8 - (char *)v5) - a3;
    if ( v36 <= 0 )
    {
LABEL_44:
      v42 = *v5 & ~(1LL << v7);
      if ( a4 )
        v42 = (1LL << v7) | *v5;
      *v5 = v42;
      result = *(unsigned int *)(a1 + 24);
      if ( (_DWORD)result == 63 )
      {
        *(_QWORD *)(a1 + 16) += 8LL;
        *(_DWORD *)(a1 + 24) = 0;
      }
      else
      {
        result = (unsigned int)(result + 1);
        *(_DWORD *)(a1 + 24) = result;
      }
      return result;
    }
    while ( 1 )
    {
      if ( v33 )
      {
        v37 = 1LL << --v33;
        if ( !v34 )
          goto LABEL_43;
      }
      else
      {
        v8 -= 8;
        v37 = 0x8000000000000000LL;
        v33 = 63;
        if ( !v34 )
        {
LABEL_43:
          --v35;
          v38 = 0x8000000000000000LL;
          v34 = 63;
          goto LABEL_38;
        }
      }
      v38 = 1LL << --v34;
LABEL_38:
      v39 = v37 & *(_QWORD *)v8;
      v40 = v38 | *v35;
      v41 = *v35 & ~v38;
      if ( v39 )
        v41 = v40;
      *v35 = v41;
      if ( !--v36 )
        goto LABEL_44;
    }
  }
  v10 = v9 + 8LL * (_QWORD)&v8[-*(_QWORD *)a1];
  if ( v10 == 0x7FFFFFFFFFFFFFC0LL )
    sub_4262D8((__int64)"vector<bool>::_M_insert_aux");
  v11 = 1;
  if ( v10 )
    v11 = v10;
  v12 = __CFADD__(v11, v10);
  v13 = v11 + v10;
  if ( v12 )
  {
    v44 = 0xFFFFFFFFFFFFFF8LL;
    v14 = (char *)sub_22077B0(0xFFFFFFFFFFFFFF8uLL);
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFC0LL )
      v13 = 0x7FFFFFFFFFFFFFC0LL;
    v44 = 8 * ((v13 + 63) >> 6);
    v14 = (char *)sub_22077B0(v44);
  }
  v15 = *(const void **)a1;
  v16 = 0;
  v43 = v14;
  if ( *(__int64 **)a1 != a2 )
  {
    memmove(v14, v15, (char *)a2 - (_BYTE *)v15);
    v16 = (char *)a2 - (_BYTE *)v15;
  }
  v17 = (__int64 *)&v43[v16];
  v18 = v7;
  v19 = *v17;
  if ( v7 )
  {
    v20 = a2;
    v21 = 0;
    do
    {
      while ( 1 )
      {
        v24 = (1LL << v21) | v19;
        v19 &= ~(1LL << v21);
        if ( ((1LL << v21) & *v20) != 0 )
          v19 = v24;
        v25 = v21 + 1;
        *v17 = v19;
        if ( v21 == 63 )
          break;
        v22 = v21 + 2;
        ++v21;
        v23 = 1LL << v25;
        if ( !--v18 )
          goto LABEL_18;
      }
      v19 = v17[1];
      ++v20;
      ++v17;
      v21 = 0;
      v22 = 1;
      v23 = 1;
      --v18;
    }
    while ( v18 );
LABEL_18:
    v26 = v17;
    if ( v21 == 63 )
    {
      v26 = v17 + 1;
      v22 = 0;
    }
  }
  else
  {
    v26 = v17;
    v23 = 1;
    v22 = 1;
  }
  v27 = v19 | v23;
  v28 = ~v23 & v19;
  if ( a4 )
    v28 = v27;
  v29 = *(unsigned int *)(a1 + 24);
  *v17 = v28;
  v30 = v29 + 8LL * (*(_QWORD *)(a1 + 16) - (_QWORD)v5) - v7;
  if ( v30 > 0 )
  {
    while ( 1 )
    {
      v31 = *v26 & ~(1LL << v22);
      if ( ((1LL << v7) & *v5) != 0 )
        v31 = (1LL << v22) | *v26;
      *v26 = v31;
      if ( v7 == 63 )
      {
        ++v5;
        v7 = 0;
        if ( v22 == 63 )
          goto LABEL_30;
LABEL_25:
        ++v22;
        if ( !--v30 )
          break;
      }
      else
      {
        ++v7;
        if ( v22 != 63 )
          goto LABEL_25;
LABEL_30:
        ++v26;
        v22 = 0;
        if ( !--v30 )
          break;
      }
    }
  }
  v45 = v22;
  if ( v15 )
  {
    j_j___libc_free_0((unsigned __int64)v15);
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_DWORD *)(a1 + 8) = 0;
  }
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v43;
  *(_QWORD *)(a1 + 32) = &v43[v44];
  *(_QWORD *)(a1 + 16) = v26;
  *(_DWORD *)(a1 + 24) = v45;
  return v45;
}
