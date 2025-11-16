// Function: sub_35360E0
// Address: 0x35360e0
//
__int64 __fastcall sub_35360E0(__int64 a1, __int64 *a2, _BYTE *a3, _BYTE *a4, _DWORD *a5, __int64 a6, __int64 a7)
{
  __int64 *v7; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rdx
  unsigned int v14; // r13d
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // r14d
  __int64 v22; // r13
  int v23; // r14d
  int v24; // eax
  __int64 v25; // r9
  __int64 v26; // rcx
  unsigned int j; // r15d
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // rsi
  bool v31; // al
  int v32; // eax
  int v33; // r13d
  __int64 v34; // r15
  unsigned int v35; // r13d
  int v36; // eax
  int v37; // r11d
  unsigned int i; // r10d
  __int64 *v39; // rcx
  bool v40; // al
  int v41; // eax
  __int64 v42; // rax
  __int64 result; // rax
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  int v49; // r15d
  __int64 v50; // r13
  int v51; // r15d
  int v52; // eax
  int v53; // ecx
  unsigned int k; // r14d
  __int64 v55; // rsi
  bool v56; // al
  unsigned int v57; // r10d
  int v58; // r15d
  unsigned int v59; // r14d
  __int64 v60; // [rsp+0h] [rbp-60h]
  unsigned int v61; // [rsp+Ch] [rbp-54h]
  int v62; // [rsp+Ch] [rbp-54h]
  int v63; // [rsp+Ch] [rbp-54h]
  __int64 v64; // [rsp+10h] [rbp-50h]
  __int64 *v65; // [rsp+10h] [rbp-50h]
  unsigned int v66; // [rsp+18h] [rbp-48h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+20h] [rbp-40h] BYREF
  int v69; // [rsp+28h] [rbp-38h]

  v7 = a2;
  *(_BYTE *)(a1 + 208) = 0;
  if ( *a3 )
    *a4 = 1;
  *a3 = 1;
  ++*a5;
  v10 = *(unsigned int *)(a7 + 8);
  v11 = *(unsigned int *)(a7 + 12);
  v12 = *a2;
  if ( v10 + 1 > v11 )
  {
    a2 = (__int64 *)(a7 + 16);
    sub_C8D5F0(a7, (const void *)(a7 + 16), v10 + 1, 8u, (__int64)a5, a6);
    v10 = *(unsigned int *)(a7 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a7 + 8 * v10) = v12;
  ++*(_DWORD *)(a7 + 8);
  v13 = *v7;
  v14 = *(_DWORD *)(a1 + 40);
  v69 = *(_DWORD *)(a1 + 12);
  v68 = v13;
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 16);
    v15 = a1 + 16;
    goto LABEL_7;
  }
  v34 = *(_QWORD *)(a1 + 24);
  v35 = v14 - 1;
  v36 = sub_2E8E920(&v68, (__int64)a2, v13, v11, (__int64)a5, a6);
  v37 = 1;
  v29 = 0;
  for ( i = v35 & v36; ; i = v35 & v57 )
  {
    v39 = (__int64 *)(v34 + 16LL * i);
    v25 = *v39;
    if ( (unsigned __int64)(v68 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v25 - 1) > 0xFFFFFFFFFFFFFFFDLL )
    {
      if ( v68 == v25 )
      {
LABEL_29:
        v33 = *((_DWORD *)v39 + 2);
        goto LABEL_30;
      }
    }
    else
    {
      v60 = v29;
      v62 = v37;
      v65 = (__int64 *)(v34 + 16LL * i);
      v66 = i;
      v40 = sub_2E88AF0(v68, *v39, 3u);
      i = v66;
      v39 = v65;
      v37 = v62;
      v29 = v60;
      if ( v40 )
        goto LABEL_29;
      v25 = *v65;
    }
    if ( !v25 )
      break;
    if ( v25 == -1 && !v29 )
      v29 = (__int64)v39;
    v57 = v37 + i;
    ++v37;
  }
  v41 = *(_DWORD *)(a1 + 32);
  v14 = *(_DWORD *)(a1 + 40);
  v15 = a1 + 16;
  if ( !v29 )
    v29 = (__int64)v39;
  ++*(_QWORD *)(a1 + 16);
  v32 = v41 + 1;
  if ( 4 * v32 >= 3 * v14 )
  {
LABEL_7:
    v16 = 2 * v14;
    sub_3535E90(v15, v16);
    v21 = *(_DWORD *)(a1 + 40);
    if ( v21 )
    {
      v22 = *(_QWORD *)(a1 + 24);
      v23 = v21 - 1;
      v24 = sub_2E8E920(&v68, v16, v17, v18, v19, v20);
      v25 = 1;
      v26 = 0;
      for ( j = v23 & v24; ; j = v23 & v58 )
      {
        v28 = v68;
        v29 = v22 + 16LL * j;
        v30 = *(_QWORD *)v29;
        if ( (unsigned __int64)(*(_QWORD *)v29 - 1LL) > 0xFFFFFFFFFFFFFFFDLL
          || (unsigned __int64)(v68 - 1) > 0xFFFFFFFFFFFFFFFDLL )
        {
          if ( v68 == v30 )
            goto LABEL_13;
        }
        else
        {
          v61 = v25;
          v64 = v26;
          v31 = sub_2E88AF0(v68, v30, 3u);
          v29 = v22 + 16LL * j;
          v26 = v64;
          v25 = v61;
          if ( v31 )
            goto LABEL_12;
          v30 = *(_QWORD *)(v22 + 16LL * j);
          v28 = v68;
        }
        if ( !v30 )
          break;
        if ( v30 != -1 || v26 )
          v29 = v26;
        v58 = v25 + j;
        v26 = v29;
        v25 = (unsigned int)(v25 + 1);
      }
      v32 = *(_DWORD *)(a1 + 32) + 1;
      if ( v26 )
        v29 = v26;
      goto LABEL_14;
    }
LABEL_65:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
  if ( v14 - (v32 + *(_DWORD *)(a1 + 36)) <= v14 >> 3 )
  {
    v44 = v14;
    sub_3535E90(v15, v14);
    v49 = *(_DWORD *)(a1 + 40);
    if ( v49 )
    {
      v50 = *(_QWORD *)(a1 + 24);
      v51 = v49 - 1;
      v52 = sub_2E8E920(&v68, v44, v45, v46, v47, v48);
      v25 = 0;
      v53 = 1;
      for ( k = v51 & v52; ; k = v51 & v59 )
      {
        v28 = v68;
        v29 = v50 + 16LL * k;
        v55 = *(_QWORD *)v29;
        if ( (unsigned __int64)(*(_QWORD *)v29 - 1LL) > 0xFFFFFFFFFFFFFFFDLL
          || (unsigned __int64)(v68 - 1) > 0xFFFFFFFFFFFFFFFDLL )
        {
          if ( v68 == v55 )
            goto LABEL_13;
        }
        else
        {
          v63 = v53;
          v67 = v25;
          v56 = sub_2E88AF0(v68, v55, 3u);
          v25 = v67;
          v29 = v50 + 16LL * k;
          v53 = v63;
          if ( v56 )
          {
LABEL_12:
            v28 = v68;
LABEL_13:
            v32 = *(_DWORD *)(a1 + 32) + 1;
            goto LABEL_14;
          }
          v55 = *(_QWORD *)(v50 + 16LL * k);
        }
        if ( !v55 )
          break;
        if ( v55 != -1 || v25 )
          v29 = v25;
        v59 = v53 + k;
        v25 = v29;
        ++v53;
      }
      v28 = v68;
      v32 = *(_DWORD *)(a1 + 32) + 1;
      if ( v25 )
        v29 = v25;
      goto LABEL_14;
    }
    goto LABEL_65;
  }
  v28 = v68;
LABEL_14:
  *(_DWORD *)(a1 + 32) = v32;
  if ( *(_QWORD *)v29 )
    --*(_DWORD *)(a1 + 36);
  *(_QWORD *)v29 = v28;
  v33 = v69;
  *(_DWORD *)(v29 + 8) = v69;
  ++*(_DWORD *)(a1 + 12);
LABEL_30:
  v42 = *(unsigned int *)(a6 + 8);
  if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, (const void *)(a6 + 16), v42 + 1, 4u, v29, v25);
    v42 = *(unsigned int *)(a6 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a6 + 4 * v42) = v33;
  ++*(_DWORD *)(a6 + 8);
  result = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) >= (unsigned int)result )
    sub_C64ED0("Instruction mapping overflow!", 1u);
  return result;
}
