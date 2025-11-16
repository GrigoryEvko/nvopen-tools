// Function: sub_302CB00
// Address: 0x302cb00
//
__int64 __fastcall sub_302CB00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v9; // rsi
  int v10; // edx
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rdi
  int v14; // r9d
  unsigned int v15; // esi
  __int64 v16; // r10
  int v17; // ebx
  unsigned int v18; // edx
  __int64 *v19; // rdi
  __int64 *v20; // rcx
  __int64 v21; // r9
  int v22; // edx
  __int64 v23; // rax
  unsigned __int8 **v24; // rbx
  unsigned __int8 *v25; // rdi
  __int64 *v26; // rbx
  __int64 *v27; // r15
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rdx
  __int64 v31; // r8
  unsigned int v32; // eax
  __int64 *v33; // rcx
  __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rdi
  int v37; // edx
  unsigned int v38; // eax
  __int64 *v39; // rsi
  __int64 v40; // r8
  __int64 v41; // rdi
  int v42; // esi
  int v43; // r9d
  int v44; // eax
  int v45; // r11d
  __int64 *v46; // r10
  int v47; // eax
  int v48; // ecx
  __int64 v49; // [rsp+0h] [rbp-70h]
  __int64 v50; // [rsp+0h] [rbp-70h]
  __int64 v51[2]; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v52; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v53; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+30h] [rbp-40h]
  __int64 v56; // [rsp+38h] [rbp-38h]

  result = a1;
  v5 = a1;
  v9 = *(_QWORD *)(a3 + 8);
  v10 = *(_DWORD *)(a3 + 24);
  v51[0] = a1;
  if ( v10 )
  {
    v11 = v10 - 1;
    v12 = (v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v13 = *(_QWORD *)(v9 + 8LL * v12);
    if ( result == v13 )
      return result;
    v14 = 1;
    while ( v13 != -4096 )
    {
      v12 = v11 & (v14 + v12);
      v13 = *(_QWORD *)(v9 + 8LL * v12);
      if ( result == v13 )
        return result;
      ++v14;
    }
  }
  v15 = *(_DWORD *)(a4 + 24);
  if ( !v15 )
  {
    ++*(_QWORD *)a4;
    v53 = 0;
LABEL_10:
    v15 *= 2;
LABEL_11:
    sub_2DD9650(a4, v15);
    sub_3028910(a4, v51, &v53);
    v5 = v51[0];
    v20 = v53;
    v22 = *(_DWORD *)(a4 + 16) + 1;
    goto LABEL_12;
  }
  v16 = *(_QWORD *)(a4 + 8);
  v17 = 1;
  v18 = (v15 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v19 = (__int64 *)(v16 + 8LL * v18);
  v20 = 0;
  v21 = *v19;
  if ( result == *v19 )
LABEL_8:
    sub_C64ED0("Circular dependency found in global variable set", 1u);
  while ( v21 != -4096 )
  {
    if ( !v20 && v21 == -8192 )
      v20 = v19;
    v18 = (v15 - 1) & (v17 + v18);
    v19 = (__int64 *)(v16 + 8LL * v18);
    v21 = *v19;
    if ( result == *v19 )
      goto LABEL_8;
    ++v17;
  }
  v44 = *(_DWORD *)(a4 + 16);
  if ( !v20 )
    v20 = v19;
  ++*(_QWORD *)a4;
  v22 = v44 + 1;
  v53 = v20;
  if ( 4 * (v44 + 1) >= 3 * v15 )
    goto LABEL_10;
  if ( v15 - *(_DWORD *)(a4 + 20) - v22 <= v15 >> 3 )
    goto LABEL_11;
LABEL_12:
  *(_DWORD *)(a4 + 16) = v22;
  if ( *v20 != -4096 )
    --*(_DWORD *)(a4 + 20);
  *v20 = v5;
  v23 = v51[0];
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  if ( (*(_BYTE *)(v51[0] + 7) & 0x40) != 0 )
  {
    v24 = *(unsigned __int8 ***)(v51[0] - 8);
    v49 = (__int64)&v24[4 * (*(_DWORD *)(v51[0] + 4) & 0x7FFFFFF)];
  }
  else
  {
    v49 = v51[0];
    v24 = (unsigned __int8 **)(v51[0] - 32LL * (*(_DWORD *)(v51[0] + 4) & 0x7FFFFFF));
  }
  if ( v24 != (unsigned __int8 **)v49 )
  {
    do
    {
      v25 = *v24;
      v24 += 4;
      sub_302C8A0(v25, (__int64)&v53);
    }
    while ( (unsigned __int8 **)v49 != v24 );
    v26 = v54;
    v27 = &v54[(unsigned int)v56];
    v23 = v51[0];
    if ( (_DWORD)v55 )
    {
      if ( v27 != v54 )
      {
        while ( *v26 == -4096 || *v26 == -8192 )
        {
          if ( v27 == ++v26 )
            goto LABEL_34;
        }
        while ( v27 != v26 )
        {
          v41 = *v26++;
          sub_302CB00(v41, a2, a3, a4);
          if ( v26 == v27 )
            break;
          while ( *v26 == -8192 || *v26 == -4096 )
          {
            if ( v27 == ++v26 )
            {
              v23 = v51[0];
              goto LABEL_19;
            }
          }
        }
LABEL_34:
        v23 = v51[0];
      }
    }
  }
LABEL_19:
  v28 = *(unsigned int *)(a2 + 8);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    v50 = v23;
    sub_C8D5F0(a2, (const void *)(a2 + 16), v28 + 1, 8u, v28 + 1, v21);
    v28 = *(unsigned int *)(a2 + 8);
    v23 = v50;
  }
  *(_QWORD *)(*(_QWORD *)a2 + 8 * v28) = v23;
  ++*(_DWORD *)(a2 + 8);
  v29 = *(_DWORD *)(a3 + 24);
  if ( !v29 )
  {
    ++*(_QWORD *)a3;
    v52 = 0;
    goto LABEL_66;
  }
  v30 = v51[0];
  v31 = *(_QWORD *)(a3 + 8);
  v32 = (v29 - 1) & ((LODWORD(v51[0]) >> 9) ^ (LODWORD(v51[0]) >> 4));
  v33 = (__int64 *)(v31 + 8LL * v32);
  v34 = *v33;
  if ( v51[0] != *v33 )
  {
    v45 = 1;
    v46 = 0;
    while ( v34 != -4096 )
    {
      if ( !v46 && v34 == -8192 )
        v46 = v33;
      v32 = (v29 - 1) & (v45 + v32);
      v33 = (__int64 *)(v31 + 8LL * v32);
      v34 = *v33;
      if ( v51[0] == *v33 )
        goto LABEL_23;
      ++v45;
    }
    v47 = *(_DWORD *)(a3 + 16);
    if ( !v46 )
      v46 = v33;
    ++*(_QWORD *)a3;
    v48 = v47 + 1;
    v52 = v46;
    if ( 4 * (v47 + 1) < 3 * v29 )
    {
      if ( v29 - *(_DWORD *)(a3 + 20) - v48 > v29 >> 3 )
      {
LABEL_62:
        *(_DWORD *)(a3 + 16) = v48;
        if ( *v46 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v46 = v30;
        goto LABEL_23;
      }
LABEL_67:
      sub_2DD9650(a3, v29);
      sub_3028910(a3, v51, &v52);
      v30 = v51[0];
      v46 = v52;
      v48 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_62;
    }
LABEL_66:
    v29 *= 2;
    goto LABEL_67;
  }
LABEL_23:
  v35 = *(_DWORD *)(a4 + 24);
  v36 = *(_QWORD *)(a4 + 8);
  if ( v35 )
  {
    v37 = v35 - 1;
    v38 = (v35 - 1) & ((LODWORD(v51[0]) >> 9) ^ (LODWORD(v51[0]) >> 4));
    v39 = (__int64 *)(v36 + 8LL * v38);
    v40 = *v39;
    if ( *v39 == v51[0] )
    {
LABEL_25:
      *v39 = -8192;
      --*(_DWORD *)(a4 + 16);
      ++*(_DWORD *)(a4 + 20);
    }
    else
    {
      v42 = 1;
      while ( v40 != -4096 )
      {
        v43 = v42 + 1;
        v38 = v37 & (v42 + v38);
        v39 = (__int64 *)(v36 + 8LL * v38);
        v40 = *v39;
        if ( v51[0] == *v39 )
          goto LABEL_25;
        v42 = v43;
      }
    }
  }
  return sub_C7D6A0((__int64)v54, 8LL * (unsigned int)v56, 8);
}
