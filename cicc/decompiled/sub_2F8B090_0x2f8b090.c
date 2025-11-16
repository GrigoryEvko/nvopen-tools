// Function: sub_2F8B090
// Address: 0x2f8b090
//
__int64 __fastcall sub_2F8B090(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 *v9; // rbx
  __int64 v10; // rax
  char **v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  char **v14; // rsi
  __int64 v15; // r12
  __int64 v16; // r13
  char **v17; // rbx
  __int64 v18; // rdx
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // r13
  char **v25; // r15
  __int64 v26; // rax
  char **v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  v6 = a3;
  v7 = a5;
  v9 = a1;
  v32 = a4;
  if ( a2 != a1 && a4 != a3 )
  {
    do
    {
      v12 = v7 + 16;
      if ( *(_DWORD *)(v6 + 8) > *((_DWORD *)v9 + 2) )
      {
        v10 = *(_QWORD *)v6;
        v11 = (char **)(v6 + 16);
        v7 += 88;
        v6 += 88;
        *(_QWORD *)(v7 - 88) = v10;
        *(_DWORD *)(v7 - 80) = *(_DWORD *)(v6 - 80);
        *(_BYTE *)(v7 - 76) = *(_BYTE *)(v6 - 76);
        sub_2F8ABB0(v12, v11, a3, a4, a5, a6);
        *(_DWORD *)(v7 - 8) = *(_DWORD *)(v6 - 8);
        if ( a2 == v9 )
          break;
      }
      else
      {
        v13 = *v9;
        v14 = (char **)(v9 + 2);
        v9 += 11;
        v7 += 88;
        *(_QWORD *)(v7 - 88) = v13;
        *(_DWORD *)(v7 - 80) = *((_DWORD *)v9 - 20);
        *(_BYTE *)(v7 - 76) = *((_BYTE *)v9 - 76);
        sub_2F8ABB0(v12, v14, a3, a4, a5, a6);
        *(_DWORD *)(v7 - 8) = *((_DWORD *)v9 - 2);
        if ( a2 == v9 )
          break;
      }
    }
    while ( v32 != v6 );
  }
  v31 = (char *)a2 - (char *)v9;
  v15 = 0x2E8BA2E8BA2E8BA3LL * (a2 - v9);
  if ( (char *)a2 - (char *)v9 > 0 )
  {
    v16 = v7 + 16;
    v17 = (char **)(v9 + 2);
    do
    {
      v18 = (__int64)*(v17 - 2);
      v19 = v17;
      v20 = v16;
      v17 += 11;
      v16 += 88;
      *(_QWORD *)(v16 - 104) = v18;
      *(_DWORD *)(v16 - 96) = *((_DWORD *)v17 - 24);
      v21 = *((unsigned __int8 *)v17 - 92);
      *(_BYTE *)(v16 - 92) = v21;
      sub_2F8ABB0(v20, v19, v21, a4, a5, a6);
      a3 = *((unsigned int *)v17 - 6);
      *(_DWORD *)(v16 - 24) = a3;
      --v15;
    }
    while ( v15 );
    a4 = v31;
    if ( v31 <= 0 )
      a4 = 88;
    v7 += a4;
  }
  v22 = v32 - v6;
  v23 = 0x2E8BA2E8BA2E8BA3LL * ((v32 - v6) >> 3);
  if ( v32 - v6 <= 0 )
    return v7;
  v24 = v7 + 16;
  v25 = (char **)(v6 + 16);
  do
  {
    v26 = (__int64)*(v25 - 2);
    v27 = v25;
    v28 = v24;
    v25 += 11;
    v24 += 88;
    *(_QWORD *)(v24 - 104) = v26;
    *(_DWORD *)(v24 - 96) = *((_DWORD *)v25 - 24);
    *(_BYTE *)(v24 - 92) = *((_BYTE *)v25 - 92);
    sub_2F8ABB0(v28, v27, a3, a4, a5, a6);
    *(_DWORD *)(v24 - 24) = *((_DWORD *)v25 - 6);
    --v23;
  }
  while ( v23 );
  v29 = 88;
  if ( v22 > 0 )
    v29 = v22;
  return v7 + v29;
}
