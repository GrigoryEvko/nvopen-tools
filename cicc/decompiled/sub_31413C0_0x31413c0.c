// Function: sub_31413C0
// Address: 0x31413c0
//
__int64 __fastcall sub_31413C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // r13
  char *v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  int v13; // eax
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r12
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rsi
  int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // r13
  __int64 v35; // rbx
  __int64 v36; // rsi
  __int64 v37; // rdi
  __int64 v39; // rdi
  char *v40; // rbx
  unsigned __int64 v41; // rbx
  __int64 v42; // rdi
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+10h] [rbp-50h] BYREF
  __int64 v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  unsigned int v47; // [rsp+28h] [rbp-38h]

  v7 = (char *)&v44;
  v8 = (char *)&v44;
  v9 = *(unsigned int *)(a1 + 24);
  v10 = *(unsigned int *)(a1 + 28);
  v44 = 0;
  v45 = 0;
  v11 = *(_QWORD *)(a1 + 16);
  v12 = v9 + 1;
  v46 = 0;
  v13 = v9;
  v47 = 0;
  if ( v9 + 1 > v10 )
  {
    v39 = a1 + 16;
    if ( v11 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v11 + 32 * v9 )
    {
      sub_3141060(v39, v12, v11, v10, a5, a6);
      v9 = *(unsigned int *)(a1 + 24);
      v11 = *(_QWORD *)(a1 + 16);
      v13 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      v40 = (char *)&v44 - v11;
      sub_3141060(v39, v12, v11, v10, a5, a6);
      v11 = *(_QWORD *)(a1 + 16);
      v9 = *(unsigned int *)(a1 + 24);
      v8 = &v40[v11];
      v13 = *(_DWORD *)(a1 + 24);
    }
  }
  v14 = v11 + 32 * v9;
  if ( v14 )
  {
    *(_QWORD *)(v14 + 16) = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_DWORD *)(v14 + 24) = 0;
    *(_QWORD *)v14 = 1;
    v15 = *((_QWORD *)v8 + 1);
    ++*(_QWORD *)v8;
    v16 = *(_QWORD *)(v14 + 8);
    *(_QWORD *)(v14 + 8) = v15;
    LODWORD(v15) = *((_DWORD *)v8 + 4);
    *((_QWORD *)v8 + 1) = v16;
    LODWORD(v16) = *(_DWORD *)(v14 + 16);
    *(_DWORD *)(v14 + 16) = v15;
    LODWORD(v15) = *((_DWORD *)v8 + 5);
    *((_DWORD *)v8 + 4) = v16;
    LODWORD(v16) = *(_DWORD *)(v14 + 20);
    *(_DWORD *)(v14 + 20) = v15;
    LODWORD(v15) = *((_DWORD *)v8 + 6);
    *((_DWORD *)v8 + 5) = v16;
    LODWORD(v16) = *(_DWORD *)(v14 + 24);
    *(_DWORD *)(v14 + 24) = v15;
    *((_DWORD *)v8 + 6) = v16;
    v17 = v47;
    v18 = v45;
    ++*(_DWORD *)(a1 + 24);
    v19 = v18;
    v43 = 72 * v17;
    v20 = v18 + 72 * v17;
    if ( (_DWORD)v17 )
    {
      do
      {
        if ( *(_QWORD *)v19 != -8192 && *(_QWORD *)v19 != -4096 )
        {
          sub_C7D6A0(*(_QWORD *)(v19 + 48), 24LL * *(unsigned int *)(v19 + 64), 8);
          sub_C7D6A0(*(_QWORD *)(v19 + 16), 24LL * *(unsigned int *)(v19 + 32), 8);
        }
        v19 += 72;
      }
      while ( v20 != v19 );
    }
    sub_C7D6A0(v18, v43, 8);
  }
  else
  {
    *(_DWORD *)(a1 + 24) = v13 + 1;
    sub_C7D6A0(0, 0, 8);
  }
  v24 = *(unsigned int *)(a1 + 104);
  v25 = *(unsigned int *)(a1 + 108);
  v44 = 0;
  v45 = 0;
  v26 = v24 + 1;
  v46 = 0;
  v27 = v24;
  v47 = 0;
  if ( v24 + 1 > v25 )
  {
    v41 = *(_QWORD *)(a1 + 96);
    v42 = a1 + 96;
    if ( v41 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v41 + 32 * v24 )
    {
      sub_3141210(v42, v26, v25, v21, v22, v23);
      v24 = *(unsigned int *)(a1 + 104);
      v28 = *(_QWORD *)(a1 + 96);
      v27 = *(_DWORD *)(a1 + 104);
    }
    else
    {
      sub_3141210(v42, v26, v25, v21, v22, v23);
      v28 = *(_QWORD *)(a1 + 96);
      v24 = *(unsigned int *)(a1 + 104);
      v7 = (char *)&v44 + v28 - v41;
      v27 = *(_DWORD *)(a1 + 104);
    }
  }
  else
  {
    v28 = *(_QWORD *)(a1 + 96);
  }
  v29 = v28 + 32 * v24;
  if ( v29 )
  {
    *(_QWORD *)(v29 + 16) = 0;
    *(_QWORD *)(v29 + 8) = 0;
    *(_DWORD *)(v29 + 24) = 0;
    *(_QWORD *)v29 = 1;
    v30 = *((_QWORD *)v7 + 1);
    ++*(_QWORD *)v7;
    v31 = *(_QWORD *)(v29 + 8);
    *(_QWORD *)(v29 + 8) = v30;
    LODWORD(v30) = *((_DWORD *)v7 + 4);
    *((_QWORD *)v7 + 1) = v31;
    LODWORD(v31) = *(_DWORD *)(v29 + 16);
    *(_DWORD *)(v29 + 16) = v30;
    LODWORD(v30) = *((_DWORD *)v7 + 5);
    *((_DWORD *)v7 + 4) = v31;
    LODWORD(v31) = *(_DWORD *)(v29 + 20);
    *(_DWORD *)(v29 + 20) = v30;
    LODWORD(v30) = *((_DWORD *)v7 + 6);
    *((_DWORD *)v7 + 5) = v31;
    LODWORD(v31) = *(_DWORD *)(v29 + 24);
    *(_DWORD *)(v29 + 24) = v30;
    *((_DWORD *)v7 + 6) = v31;
    v32 = v47;
    ++*(_DWORD *)(a1 + 104);
    v29 = v45;
    v33 = 48 * v32;
    if ( (_DWORD)v32 )
    {
      v34 = v45 + v33;
      v35 = v45;
      do
      {
        while ( *(_QWORD *)v35 == -1 || *(_QWORD *)v35 == -2 )
        {
          v35 += 48;
          if ( v34 == v35 )
            return sub_C7D6A0(v29, v33, 8);
        }
        v36 = *(unsigned int *)(v35 + 40);
        v37 = *(_QWORD *)(v35 + 24);
        v35 += 48;
        sub_C7D6A0(v37, 32 * v36, 8);
      }
      while ( v34 != v35 );
    }
  }
  else
  {
    v33 = 0;
    *(_DWORD *)(a1 + 104) = v27 + 1;
  }
  return sub_C7D6A0(v29, v33, 8);
}
