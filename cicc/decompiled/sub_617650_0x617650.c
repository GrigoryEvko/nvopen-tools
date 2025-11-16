// Function: sub_617650
// Address: 0x617650
//
__int64 __fastcall sub_617650(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v6; // eax
  int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 *v10; // r8
  __int64 v11; // rax
  unsigned int v12; // ecx
  int v13; // eax
  __int64 result; // rax
  __int64 *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // ecx
  int v20; // eax
  unsigned int v21; // r12d
  unsigned int v22; // r15d
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // r14
  __int64 *v28; // r8
  __int64 *v29; // r13
  int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // ecx
  __int64 *j; // rax
  __int64 v34; // r13
  _QWORD *v35; // rax
  __int64 *v36; // r13
  int v37; // eax
  __int64 v38; // rax
  unsigned int v39; // ecx
  __int64 *i; // rax
  __int64 *v41; // [rsp+0h] [rbp-50h]
  __int64 v42; // [rsp+0h] [rbp-50h]
  unsigned int v43; // [rsp+8h] [rbp-48h]
  __int64 *v44; // [rsp+8h] [rbp-48h]
  unsigned int v45; // [rsp+8h] [rbp-48h]
  __int64 *v46; // [rsp+8h] [rbp-48h]

  v6 = sub_887620(a2);
  v7 = *(_DWORD *)(a1 + 8);
  v8 = v7 & v6;
  v9 = 16 * v8;
  v10 = (__int64 *)(*(_QWORD *)a1 + 16 * v8);
  if ( !*v10 )
  {
    v11 = *a3;
    *v10 = a2;
    if ( a2 )
      v10[1] = v11;
    v12 = *(_DWORD *)(a1 + 8);
    v13 = *(_DWORD *)(a1 + 12) + 1;
    *(_DWORD *)(a1 + 12) = v13;
    result = (unsigned int)(2 * v13);
    if ( (unsigned int)result <= v12 )
      return result;
    v45 = v12;
    v21 = v12 + 1;
    v22 = 2 * v12 + 1;
    v34 = 2 * v12 + 2;
    v35 = (_QWORD *)sub_822B10(16 * v34);
    v26 = v45;
    v27 = v35;
    if ( (_DWORD)v34 )
    {
      v25 = (__int64)&v35[2 * v22 + 2];
      do
      {
        if ( v35 )
          *v35 = 0;
        v35 += 2;
      }
      while ( (_QWORD *)v25 != v35 );
    }
    v28 = *(__int64 **)a1;
    if ( v21 )
    {
      v26 = 16LL * v45;
      v36 = *(__int64 **)a1;
      v25 = (__int64)v28 + v26 + 16;
      do
      {
        if ( *v36 )
        {
          v42 = v25;
          v46 = v28;
          v37 = sub_887620(*v36);
          v28 = v46;
          v25 = v42;
          v38 = v22 & v37;
          v39 = v38;
          for ( i = &v27[2 * v38]; *i; i = &v27[2 * v39] )
            v39 = v22 & (v39 + 1);
          v26 = *v36;
          *i = *v36;
          if ( v26 )
          {
            v26 = v36[1];
            i[1] = v26;
          }
        }
        v36 += 2;
      }
      while ( v36 != (__int64 *)v25 );
    }
LABEL_25:
    *(_QWORD *)a1 = v27;
    *(_DWORD *)(a1 + 8) = v22;
    return sub_822B90(v28, 16LL * v21, v25, v26);
  }
  do
  {
    LODWORD(v8) = v7 & (v8 + 1);
    v15 = (__int64 *)(*(_QWORD *)a1 + 16LL * (unsigned int)v8);
  }
  while ( *v15 );
  v16 = *v10;
  *v15 = *v10;
  if ( v16 )
    v15[1] = v10[1];
  *v10 = 0;
  v17 = (_QWORD *)(*(_QWORD *)a1 + v9);
  v18 = *a3;
  *v17 = a2;
  if ( a2 )
    v17[1] = v18;
  v19 = *(_DWORD *)(a1 + 8);
  v20 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v20;
  result = (unsigned int)(2 * v20);
  if ( (unsigned int)result > v19 )
  {
    v43 = v19;
    v21 = v19 + 1;
    v22 = 2 * v19 + 1;
    v23 = 2 * v19 + 2;
    v24 = (_QWORD *)sub_822B10(16 * v23);
    v26 = v43;
    v27 = v24;
    if ( (_DWORD)v23 )
    {
      v25 = (__int64)&v24[2 * v22 + 2];
      do
      {
        if ( v24 )
          *v24 = 0;
        v24 += 2;
      }
      while ( (_QWORD *)v25 != v24 );
    }
    v28 = *(__int64 **)a1;
    if ( v21 )
    {
      v26 = 16LL * v43;
      v29 = *(__int64 **)a1;
      v25 = (__int64)v28 + v26 + 16;
      do
      {
        while ( 1 )
        {
          if ( *v29 )
          {
            v41 = v28;
            v44 = (__int64 *)v25;
            v30 = sub_887620(*v29);
            v25 = (__int64)v44;
            v28 = v41;
            v31 = v22 & v30;
            v32 = v31;
            for ( j = &v27[2 * v31]; *j; j = &v27[2 * v32] )
              v32 = v22 & (v32 + 1);
            v26 = *v29;
            *j = *v29;
            if ( v26 )
              break;
          }
          v29 += 2;
          if ( (__int64 *)v25 == v29 )
            goto LABEL_25;
        }
        v26 = v29[1];
        v29 += 2;
        j[1] = v26;
      }
      while ( v44 != v29 );
    }
    goto LABEL_25;
  }
  return result;
}
