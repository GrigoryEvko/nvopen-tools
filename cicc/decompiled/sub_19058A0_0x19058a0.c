// Function: sub_19058A0
// Address: 0x19058a0
//
__int64 __fastcall sub_19058A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // rdi
  int v15; // edx
  unsigned int v16; // eax
  unsigned int v17; // edx
  unsigned int v18; // esi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  unsigned int v22; // edi
  __int64 v23; // rcx
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // edx
  __int64 v29; // rbx
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rsi
  int v33; // r9d
  int v34; // r10d
  int v35; // eax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rsi
  int v39; // r8d
  unsigned int v40; // r13d
  __int64 v41; // rdi
  __int64 v42; // rcx
  int v43; // r9d
  __int64 v44; // r8
  __int64 v45; // [rsp+0h] [rbp-50h] BYREF
  __int64 v46; // [rsp+8h] [rbp-48h]
  unsigned int v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+18h] [rbp-38h]
  unsigned int v49; // [rsp+20h] [rbp-30h]

  v6 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v7 + 16 * v6) )
      {
        v11 = *(_QWORD *)(a1 + 32) + 40LL * *((unsigned int *)v9 + 2);
        if ( *(_QWORD *)(a1 + 40) != v11 )
        {
          if ( *(_DWORD *)(v11 + 16) > 0x40u )
          {
            v12 = *(_QWORD *)(v11 + 8);
            if ( v12 )
              j_j___libc_free_0_0(v12);
          }
          *(_QWORD *)(v11 + 8) = *a3;
          *(_DWORD *)(v11 + 16) = *((_DWORD *)a3 + 2);
          *((_DWORD *)a3 + 2) = 0;
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
          {
            v14 = *(_QWORD *)(v11 + 24);
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
          *(_QWORD *)(v11 + 24) = a3[2];
          result = *((unsigned int *)a3 + 6);
          *(_DWORD *)(v11 + 32) = result;
          *((_DWORD *)a3 + 6) = 0;
          return result;
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v10 != -8 )
      {
        v33 = v15 + 1;
        v8 = (v6 - 1) & (v15 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_3;
        v15 = v33;
      }
    }
  }
  v16 = *((_DWORD *)a3 + 2);
  v17 = *((_DWORD *)a3 + 6);
  *((_DWORD *)a3 + 2) = 0;
  *((_DWORD *)a3 + 6) = 0;
  v18 = *(_DWORD *)(a1 + 24);
  v47 = v16;
  v19 = *a3;
  v45 = a2;
  v46 = v19;
  v20 = a3[2];
  v49 = v17;
  v48 = v20;
  if ( !v18 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_24;
  }
  v21 = *(_QWORD *)(a1 + 8);
  v22 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v21 + 16LL * v22;
  v23 = *(_QWORD *)result;
  if ( *(_QWORD *)result != a2 )
  {
    v34 = 1;
    v29 = 0;
    while ( v23 != -8 )
    {
      if ( v29 || v23 != -16 )
        result = v29;
      v22 = (v18 - 1) & (v34 + v22);
      v23 = *(_QWORD *)(v21 + 16LL * v22);
      if ( v23 == a2 )
        goto LABEL_17;
      ++v34;
      v29 = result;
      result = v21 + 16LL * v22;
    }
    if ( !v29 )
      v29 = result;
    v35 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v28 = v35 + 1;
    if ( 4 * (v35 + 1) < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(a1 + 20) - v28 > v18 >> 3 )
      {
LABEL_26:
        *(_DWORD *)(a1 + 16) = v28;
        if ( *(_QWORD *)v29 != -8 )
          --*(_DWORD *)(a1 + 20);
        *(_QWORD *)v29 = a2;
        *(_DWORD *)(v29 + 8) = 0;
        v31 = *(_QWORD *)(a1 + 40);
        if ( v31 == *(_QWORD *)(a1 + 48) )
        {
          sub_1905440((__int64 *)(a1 + 32), v31, (__int64)&v45);
          v32 = *(_QWORD *)(a1 + 40);
        }
        else
        {
          if ( v31 )
          {
            *(_QWORD *)v31 = v45;
            *(_DWORD *)(v31 + 16) = v47;
            *(_QWORD *)(v31 + 8) = v46;
            v47 = 0;
            *(_DWORD *)(v31 + 32) = v49;
            *(_QWORD *)(v31 + 24) = v48;
            v31 = *(_QWORD *)(a1 + 40);
            v49 = 0;
          }
          v32 = v31 + 40;
          *(_QWORD *)(a1 + 40) = v32;
        }
        result = -858993459 * (unsigned int)((v32 - *(_QWORD *)(a1 + 32)) >> 3) - 1;
        *(_DWORD *)(v29 + 8) = result;
        v17 = v49;
        goto LABEL_17;
      }
      sub_14672C0(a1, v18);
      v36 = *(_DWORD *)(a1 + 24);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 8);
        v39 = 1;
        v40 = v37 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v28 = *(_DWORD *)(a1 + 16) + 1;
        v41 = 0;
        v29 = v38 + 16LL * v40;
        v42 = *(_QWORD *)v29;
        if ( *(_QWORD *)v29 != a2 )
        {
          while ( v42 != -8 )
          {
            if ( !v41 && v42 == -16 )
              v41 = v29;
            v40 = v37 & (v39 + v40);
            v29 = v38 + 16LL * v40;
            v42 = *(_QWORD *)v29;
            if ( *(_QWORD *)v29 == a2 )
              goto LABEL_26;
            ++v39;
          }
          if ( v41 )
            v29 = v41;
        }
        goto LABEL_26;
      }
LABEL_68:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_24:
    sub_14672C0(a1, 2 * v18);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v29 = v26 + 16LL * v27;
      v30 = *(_QWORD *)v29;
      if ( *(_QWORD *)v29 != a2 )
      {
        v43 = 1;
        v44 = 0;
        while ( v30 != -8 )
        {
          if ( v30 == -16 && !v44 )
            v44 = v29;
          v27 = v25 & (v43 + v27);
          v29 = v26 + 16LL * v27;
          v30 = *(_QWORD *)v29;
          if ( *(_QWORD *)v29 == a2 )
            goto LABEL_26;
          ++v43;
        }
        if ( v44 )
          v29 = v44;
      }
      goto LABEL_26;
    }
    goto LABEL_68;
  }
LABEL_17:
  if ( v17 > 0x40 && v48 )
    result = j_j___libc_free_0_0(v48);
  if ( v47 > 0x40 && v46 )
    return j_j___libc_free_0_0(v46);
  return result;
}
