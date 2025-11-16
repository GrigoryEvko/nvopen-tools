// Function: sub_2F8C830
// Address: 0x2f8c830
//
unsigned __int64 __fastcall sub_2F8C830(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 v15; // rcx
  unsigned __int64 result; // rax
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r12
  _DWORD *v22; // r13
  __int64 v23; // rbx
  int v24; // edx
  char **v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rbx
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // r12d
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // r12d
  __int64 v42; // rbx
  unsigned __int64 v43; // r15
  __int64 v44; // r13
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  int v49; // r12d
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // r12d
  unsigned __int64 v55; // [rsp+0h] [rbp-50h]
  unsigned __int64 v56; // [rsp+8h] [rbp-48h]
  __int64 v57; // [rsp+8h] [rbp-48h]
  __int64 v58; // [rsp+8h] [rbp-48h]
  unsigned __int64 v59[7]; // [rsp+18h] [rbp-38h] BYREF

  v56 = a2;
  v9 = *(_QWORD *)a1;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(unsigned int *)(a1 + 12);
  v12 = 5 * v10;
  v13 = v10;
  v14 = v10 + 1;
  v15 = 16 * v12;
  result = v9 + v15;
  if ( v9 + v15 == a2 )
  {
    if ( v14 > v11 )
    {
      v42 = a1 + 16;
      if ( v9 > a3 || result <= a3 )
      {
        v44 = sub_C8D7D0(a1, a1 + 16, v14, 0x50u, v59, a6);
        sub_2F8C750((__int64 *)a1, v44, v50, v51, v52, v53);
        v54 = v59[0];
        if ( *(_QWORD *)a1 != v42 )
          _libc_free(*(_QWORD *)a1);
        *(_QWORD *)a1 = v44;
        *(_DWORD *)(a1 + 12) = v54;
      }
      else
      {
        v43 = a3 - v9;
        v44 = sub_C8D7D0(a1, a1 + 16, v14, 0x50u, v59, a6);
        sub_2F8C750((__int64 *)a1, v44, v45, v46, v47, v48);
        v49 = v59[0];
        if ( *(_QWORD *)a1 != v42 )
          _libc_free(*(_QWORD *)a1);
        *(_QWORD *)a1 = v44;
        a3 = v44 + v43;
        *(_DWORD *)(a1 + 12) = v49;
      }
      v13 = *(unsigned int *)(a1 + 8);
      result = v44 + 80 * v13;
      v56 = result;
    }
    if ( v56 )
    {
      *(_QWORD *)v56 = *(_QWORD *)a3;
      *(_QWORD *)(v56 + 8) = v56 + 24;
      *(_QWORD *)(v56 + 16) = 0x600000000LL;
      if ( *(_DWORD *)(a3 + 16) )
        sub_2F8AAD0(v56 + 8, a3 + 8, v13, v56, v11, a6);
      result = *(unsigned int *)(a3 + 72);
      *(_DWORD *)(v56 + 72) = result;
      LODWORD(v13) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v13 + 1;
  }
  else
  {
    if ( v14 > v11 )
    {
      v29 = a1 + 16;
      v30 = v56 - v9;
      if ( v9 > a3 || result <= a3 )
      {
        v58 = sub_C8D7D0(a1, a1 + 16, v14, 0x50u, v59, a6);
        sub_2F8C750((__int64 *)a1, v58, v37, v38, v39, v40);
        v41 = v59[0];
        v9 = v58;
        if ( *(_QWORD *)a1 != v29 )
        {
          _libc_free(*(_QWORD *)a1);
          v9 = v58;
        }
        *(_QWORD *)a1 = v9;
        *(_DWORD *)(a1 + 12) = v41;
      }
      else
      {
        v31 = a3 - v9;
        v57 = sub_C8D7D0(a1, a1 + 16, v14, 0x50u, v59, a6);
        sub_2F8C750((__int64 *)a1, v57, v32, v33, v34, v35);
        v36 = v59[0];
        v9 = v57;
        if ( *(_QWORD *)a1 != v29 )
        {
          _libc_free(*(_QWORD *)a1);
          v9 = v57;
        }
        *(_QWORD *)a1 = v9;
        a3 = v9 + v31;
        *(_DWORD *)(a1 + 12) = v36;
      }
      v56 = v9 + v30;
      LODWORD(v13) = *(_DWORD *)(a1 + 8);
      v15 = 80LL * (unsigned int)v13;
      result = v9 + v15;
    }
    v17 = v9 + v15 - 80;
    if ( result )
    {
      v18 = *(_QWORD *)v17;
      *(_QWORD *)(result + 16) = 0x600000000LL;
      *(_QWORD *)result = v18;
      *(_QWORD *)(result + 8) = result + 24;
      v19 = *(unsigned int *)(v17 + 16);
      if ( (_DWORD)v19 )
      {
        v55 = result;
        sub_2F8ABB0(result + 8, (char **)(v17 + 8), v19, 0x600000000LL, v11, a6);
        result = v55;
      }
      *(_DWORD *)(result + 72) = *(_DWORD *)(v17 + 72);
      v9 = *(_QWORD *)a1;
      LODWORD(v13) = *(_DWORD *)(a1 + 8);
      result = *(_QWORD *)a1 + 80LL * (unsigned int)v13;
      v17 = result - 80;
    }
    v20 = v17 - v56;
    v21 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v17 - v56) >> 4);
    if ( (__int64)(v17 - v56) > 0 )
    {
      v22 = (_DWORD *)(result - 72);
      v23 = v17 - 72;
      do
      {
        v24 = *(_DWORD *)(v23 - 8);
        v25 = (char **)v23;
        v26 = (__int64)v22;
        v23 -= 80;
        v22 -= 20;
        v22[18] = v24;
        v27 = *(unsigned int *)(v23 + 76);
        v22[19] = v27;
        sub_2F8ABB0(v26, v25, v27, v20, v11, a6);
        v22[36] = *(_DWORD *)(v23 + 144);
        --v21;
      }
      while ( v21 );
      LODWORD(v13) = *(_DWORD *)(a1 + 8);
      v9 = *(_QWORD *)a1;
    }
    v28 = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 8) = v28;
    if ( v56 <= a3 && a3 < 80 * v28 + v9 )
      a3 += 80LL;
    *(_DWORD *)v56 = *(_DWORD *)a3;
    *(_DWORD *)(v56 + 4) = *(_DWORD *)(a3 + 4);
    sub_2F8AAD0(v56 + 8, a3 + 8, v28, v20, v11, a6);
    result = *(unsigned int *)(a3 + 72);
    *(_DWORD *)(v56 + 72) = result;
  }
  return result;
}
