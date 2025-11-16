// Function: sub_FF1FB0
// Address: 0xff1fb0
//
__int64 __fastcall sub_FF1FB0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // ebx
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  unsigned int v19; // esi
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 result; // rax
  __int64 v23; // rdx
  int v24; // r8d
  __int64 v25; // rdi
  int v26; // ecx
  int v27; // edx
  unsigned __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // r8d
  int v39; // r8d
  __int64 v40; // r9
  __int64 v41; // rdi
  __int64 v42; // rbx
  int v43; // ecx
  __int64 v44; // rsi
  int v45; // r9d
  int v46; // r9d
  __int64 v47; // r10
  unsigned int v48; // ecx
  __int64 v49; // r8
  int v50; // edi
  __int64 v51; // rsi
  int v53; // [rsp+10h] [rbp-40h]
  unsigned int v54; // [rsp+14h] [rbp-3Ch]
  int v55; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 16);
  if ( v5 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v5 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_7;
    }
LABEL_5:
    if ( (unsigned int)sub_FEEEB0(a1, *(_QWORD *)(v6 + 40)) != a3 )
    {
      v54 = 1;
      goto LABEL_8;
    }
    while ( 1 )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        break;
      v6 = *(_QWORD *)(v5 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
        goto LABEL_5;
    }
  }
LABEL_7:
  v54 = 0;
LABEL_8:
  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == a2 + 48 )
    goto LABEL_34;
  if ( !v7 )
    BUG();
  v8 = v7 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
    goto LABEL_34;
  v53 = sub_B46E30(v8);
  v55 = v53 >> 2;
  if ( v53 >> 2 <= 0 )
  {
    v34 = v53;
    v9 = 0;
LABEL_45:
    if ( v34 != 2 )
    {
      if ( v34 != 3 )
      {
        if ( v34 != 1 )
          goto LABEL_34;
LABEL_52:
        v37 = sub_B46EC0(v8, v9);
        if ( (unsigned int)sub_FEEEB0(a1, v37) == a3 )
          goto LABEL_34;
        goto LABEL_18;
      }
      v35 = sub_B46EC0(v8, v9);
      if ( (unsigned int)sub_FEEEB0(a1, v35) != a3 )
      {
LABEL_18:
        if ( v9 == v53 )
          goto LABEL_34;
        goto LABEL_19;
      }
      ++v9;
    }
    v36 = sub_B46EC0(v8, v9);
    if ( (unsigned int)sub_FEEEB0(a1, v36) == a3 )
    {
      ++v9;
      goto LABEL_52;
    }
    goto LABEL_18;
  }
  v9 = 0;
  while ( 1 )
  {
    v14 = sub_B46EC0(v8, v9);
    if ( (unsigned int)sub_FEEEB0(a1, v14) != a3 )
      goto LABEL_18;
    v10 = v9 + 1;
    v11 = sub_B46EC0(v8, v9 + 1);
    if ( (unsigned int)sub_FEEEB0(a1, v11) != a3 )
      break;
    v10 = v9 + 2;
    v12 = sub_B46EC0(v8, v9 + 2);
    if ( (unsigned int)sub_FEEEB0(a1, v12) != a3 )
      break;
    v10 = v9 + 3;
    v13 = sub_B46EC0(v8, v9 + 3);
    if ( (unsigned int)sub_FEEEB0(a1, v13) != a3 )
      break;
    v9 += 4;
    if ( !--v55 )
    {
      v34 = v53 - v9;
      goto LABEL_45;
    }
  }
  if ( v10 == v53 )
  {
LABEL_34:
    v15 = *(_QWORD *)(a1 + 40);
    v16 = *(_QWORD *)(a1 + 32);
    v17 = (v15 - v16) >> 5;
    if ( a3 < v17 )
      goto LABEL_35;
    goto LABEL_37;
  }
LABEL_19:
  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(_QWORD *)(a1 + 32);
  v54 |= 2u;
  v17 = (v15 - v16) >> 5;
  if ( a3 < v17 )
    goto LABEL_20;
LABEL_37:
  v28 = (int)(a3 + 1);
  if ( v28 > v17 )
  {
    sub_FF1200((__int64 *)(a1 + 32), v28 - v17);
    v16 = *(_QWORD *)(a1 + 32);
  }
  else if ( v28 < v17 )
  {
    v29 = 32 * v28;
    v30 = v16 + v29;
    if ( v15 != v16 + v29 )
    {
      v31 = v16 + v29;
      do
      {
        v32 = *(unsigned int *)(v31 + 24);
        v33 = *(_QWORD *)(v31 + 8);
        v31 += 32;
        sub_C7D6A0(v33, 16 * v32, 8);
      }
      while ( v15 != v31 );
      *(_QWORD *)(a1 + 40) = v30;
      v16 = *(_QWORD *)(a1 + 32);
    }
  }
LABEL_35:
  result = v54;
  if ( !v54 )
    return result;
LABEL_20:
  v18 = v16 + 32LL * (int)a3;
  v19 = *(_DWORD *)(v18 + 24);
  if ( !v19 )
  {
    ++*(_QWORD *)v18;
    goto LABEL_63;
  }
  v20 = *(_QWORD *)(v18 + 8);
  v21 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v20 + 16LL * v21;
  v23 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    return result;
  v24 = 1;
  v25 = 0;
  while ( v23 != -4096 )
  {
    if ( v23 == -8192 && !v25 )
      v25 = result;
    v21 = (v19 - 1) & (v24 + v21);
    result = v20 + 16LL * v21;
    v23 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      return result;
    ++v24;
  }
  v26 = *(_DWORD *)(v18 + 16);
  if ( v25 )
    result = v25;
  ++*(_QWORD *)v18;
  v27 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v19 )
  {
LABEL_63:
    sub_A4A350(v18, 2 * v19);
    v45 = *(_DWORD *)(v18 + 24);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(v18 + 8);
      v27 = *(_DWORD *)(v18 + 16) + 1;
      v48 = v46 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = v47 + 16LL * v48;
      v49 = *(_QWORD *)result;
      if ( a2 != *(_QWORD *)result )
      {
        v50 = 1;
        v51 = 0;
        while ( v49 != -4096 )
        {
          if ( !v51 && v49 == -8192 )
            v51 = result;
          v48 = v46 & (v50 + v48);
          result = v47 + 16LL * v48;
          v49 = *(_QWORD *)result;
          if ( a2 == *(_QWORD *)result )
            goto LABEL_28;
          ++v50;
        }
        if ( v51 )
          result = v51;
      }
      goto LABEL_28;
    }
    goto LABEL_86;
  }
  if ( v19 - *(_DWORD *)(v18 + 20) - v27 <= v19 >> 3 )
  {
    sub_A4A350(v18, v19);
    v38 = *(_DWORD *)(v18 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(v18 + 8);
      v41 = 0;
      LODWORD(v42) = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(v18 + 16) + 1;
      v43 = 1;
      result = v40 + 16LL * (unsigned int)v42;
      v44 = *(_QWORD *)result;
      if ( a2 != *(_QWORD *)result )
      {
        while ( v44 != -4096 )
        {
          if ( !v41 && v44 == -8192 )
            v41 = result;
          v42 = v39 & (unsigned int)(v42 + v43);
          result = v40 + 16 * v42;
          v44 = *(_QWORD *)result;
          if ( a2 == *(_QWORD *)result )
            goto LABEL_28;
          ++v43;
        }
        if ( v41 )
          result = v41;
      }
      goto LABEL_28;
    }
LABEL_86:
    ++*(_DWORD *)(v18 + 16);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(v18 + 16) = v27;
  if ( *(_QWORD *)result != -4096 )
    --*(_DWORD *)(v18 + 20);
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 8) = v54;
  return result;
}
