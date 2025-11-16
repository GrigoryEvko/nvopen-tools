// Function: sub_1DD2790
// Address: 0x1dd2790
//
__int64 __fastcall sub_1DD2790(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, char a5, unsigned int *a6)
{
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 result; // rax
  char v16; // [rsp+4h] [rbp-2Ch]
  __int64 v17; // [rsp+8h] [rbp-28h]

  if ( a5 )
    *a4 += *(_QWORD *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 8);
  v9 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 16);
  v8 = v9;
  if ( *a6 >= v9 )
    v9 = *a6;
  *a6 = v9;
  v10 = v8 * ((v8 + *a4 - 1) / v8);
  *a4 = v10;
  if ( a5 )
    v10 = -v10;
  *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8LL * (int)a3) = v10;
  v11 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 124) )
  {
    v16 = a5;
    v17 = v10;
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 16, a5, (int)a6);
    v11 = *(unsigned int *)(a2 + 120);
    a5 = v16;
    v10 = v17;
  }
  v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 16 * v11);
  v12[1] = v10;
  *v12 = a3;
  v13 = *(_QWORD *)(a2 + 8);
  v14 = *(_DWORD *)(a2 + 32) + a3;
  ++*(_DWORD *)(a2 + 120);
  result = v13 + 40 * v14;
  *(_BYTE *)(result + 32) = 1;
  if ( !a5 )
  {
    result = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 8);
    *a4 += result;
  }
  return result;
}
