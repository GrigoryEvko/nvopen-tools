// Function: sub_2DD15F0
// Address: 0x2dd15f0
//
__int64 __fastcall sub_2DD15F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 result; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // r13
  unsigned __int64 *v15; // rsi
  unsigned int v16; // esi
  int v17; // r15d
  __int64 v18; // r8
  __int64 *v19; // r11
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  int v24; // r9d
  int v25; // edi
  int v26; // ecx
  int v27; // edx
  int v28; // edx
  __int64 v29; // r8
  unsigned int v30; // esi
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // edx
  int v35; // esi
  __int64 v36; // rdi
  __int64 *v37; // r8
  unsigned int v38; // r13d
  int v39; // r9d
  __int64 v40; // rdx
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(unsigned int *)(a1 + 272);
  v5 = *(_QWORD *)(a1 + 256);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v24 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v24;
      }
    }
  }
  v11 = sub_B2DBE0(a2);
  v12 = sub_2DD0F80(a1, *(_BYTE **)v11, *(_QWORD *)(v11 + 8));
  v13 = (_QWORD *)sub_22077B0(0x48u);
  v14 = v13;
  if ( v13 )
    sub_2DD0370(v13, a2, v12);
  v43[0] = (__int64)v14;
  v15 = *(unsigned __int64 **)(a1 + 232);
  if ( v15 == *(unsigned __int64 **)(a1 + 240) )
  {
    sub_2DD0CA0((unsigned __int64 *)(a1 + 224), v15, v43);
    v14 = (_QWORD *)v43[0];
  }
  else
  {
    if ( v15 )
    {
      *v15 = (unsigned __int64)v14;
      *(_QWORD *)(a1 + 232) += 8LL;
      goto LABEL_12;
    }
    *(_QWORD *)(a1 + 232) = 8;
  }
  if ( v14 )
  {
    sub_2DD06B0(v14);
    j_j___libc_free_0((unsigned __int64)v14);
  }
LABEL_12:
  v16 = *(_DWORD *)(a1 + 272);
  result = *(_QWORD *)(*(_QWORD *)(a1 + 232) - 8LL);
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 248);
    goto LABEL_36;
  }
  v17 = 1;
  v18 = *(_QWORD *)(a1 + 256);
  v19 = 0;
  v20 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v21 = (__int64 *)(v18 + 16LL * v20);
  v22 = *v21;
  if ( a2 != *v21 )
  {
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v19 )
        v19 = v21;
      v20 = (v16 - 1) & (v17 + v20);
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( a2 == *v21 )
        goto LABEL_14;
      ++v17;
    }
    v25 = *(_DWORD *)(a1 + 264);
    if ( !v19 )
      v19 = v21;
    ++*(_QWORD *)(a1 + 248);
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(a1 + 268) - v26 > v16 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(a1 + 264) = v26;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 268);
        *v19 = a2;
        v23 = v19 + 1;
        v19[1] = 0;
        goto LABEL_15;
      }
      v42 = result;
      sub_2DD1410(a1 + 248, v16);
      v34 = *(_DWORD *)(a1 + 272);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 256);
        v37 = 0;
        v38 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v39 = 1;
        v26 = *(_DWORD *)(a1 + 264) + 1;
        result = v42;
        v19 = (__int64 *)(v36 + 16LL * v38);
        v40 = *v19;
        if ( a2 != *v19 )
        {
          while ( v40 != -4096 )
          {
            if ( !v37 && v40 == -8192 )
              v37 = v19;
            v38 = v35 & (v39 + v38);
            v19 = (__int64 *)(v36 + 16LL * v38);
            v40 = *v19;
            if ( a2 == *v19 )
              goto LABEL_28;
            ++v39;
          }
          if ( v37 )
            v19 = v37;
        }
        goto LABEL_28;
      }
LABEL_59:
      ++*(_DWORD *)(a1 + 264);
      BUG();
    }
LABEL_36:
    v41 = result;
    sub_2DD1410(a1 + 248, 2 * v16);
    v27 = *(_DWORD *)(a1 + 272);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 256);
      v30 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(a1 + 264) + 1;
      result = v41;
      v19 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v19;
      if ( a2 != *v19 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -4096 )
        {
          if ( v31 == -8192 && !v33 )
            v33 = v19;
          v30 = v28 & (v32 + v30);
          v19 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v19;
          if ( a2 == *v19 )
            goto LABEL_28;
          ++v32;
        }
        if ( v33 )
          v19 = v33;
      }
      goto LABEL_28;
    }
    goto LABEL_59;
  }
LABEL_14:
  v23 = v21 + 1;
LABEL_15:
  *v23 = result;
  return result;
}
