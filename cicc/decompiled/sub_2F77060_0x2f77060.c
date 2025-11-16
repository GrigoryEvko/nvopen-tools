// Function: sub_2F77060
// Address: 0x2f77060
//
__int64 __fastcall sub_2F77060(__int64 a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int *v4; // rbx
  unsigned int *v6; // r12
  unsigned int *v7; // r14
  unsigned int v8; // esi
  unsigned int v9; // edi
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 v18; // rsi
  unsigned int v19; // edi
  __int64 v20; // rcx
  unsigned int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // rdx

  result = 3 * a3;
  v4 = &a2[6 * a3];
  if ( v4 != a2 )
  {
    v6 = a2;
    v7 = a2;
    do
    {
      v8 = *v7;
      v9 = *v7;
      if ( (*v7 & 0x80000000) != 0 )
        v9 = *(_DWORD *)(a1 + 320) + (v9 & 0x7FFFFFFF);
      v10 = *(unsigned int *)(a1 + 104);
      v11 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v9);
      if ( v11 >= (unsigned int)v10 )
        goto LABEL_22;
      v12 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v13 = v12 + 24LL * v11;
        if ( v9 == *(_DWORD *)v13 )
          break;
        v11 += 256;
        if ( (unsigned int)v10 <= v11 )
          goto LABEL_22;
      }
      if ( v13 == v12 + 24 * v10 )
      {
LABEL_22:
        v15 = 0;
        v14 = 0;
      }
      else
      {
        v14 = *(_QWORD *)(v13 + 8);
        v15 = *(_QWORD *)(v13 + 16);
      }
      v16 = *((_QWORD *)v7 + 1);
      v17 = *((_QWORD *)v7 + 2);
      v7 += 6;
      sub_2F74DB0(a1, v8, v14, v15, v14 | v16, v15 | v17);
    }
    while ( v4 != v7 );
    do
    {
      v18 = *v6;
      v19 = v18;
      if ( (v18 & 0x80000000) != 0LL )
        v19 = *(_DWORD *)(a1 + 320) + (v18 & 0x7FFFFFFF);
      v20 = *(unsigned int *)(a1 + 104);
      v21 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 304) + v19);
      if ( v21 >= (unsigned int)v20 )
        goto LABEL_23;
      v22 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v23 = v22 + 24LL * v21;
        if ( v19 == *(_DWORD *)v23 )
          break;
        v21 += 256;
        if ( (unsigned int)v20 <= v21 )
          goto LABEL_23;
      }
      if ( v23 == v22 + 24 * v20 )
      {
LABEL_23:
        v25 = 0;
        v24 = 0;
      }
      else
      {
        v24 = *(_QWORD *)(v23 + 8);
        v25 = *(_QWORD *)(v23 + 16);
      }
      v26 = *((_QWORD *)v6 + 2);
      v27 = *((_QWORD *)v6 + 1);
      v6 += 6;
      result = sub_2F74F40(a1, v18, v24 | v27, v25 | v26, v24, v25);
    }
    while ( v4 != v6 );
  }
  return result;
}
