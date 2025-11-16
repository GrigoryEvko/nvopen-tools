// Function: sub_1ECBE90
// Address: 0x1ecbe90
//
__int64 __fastcall sub_1ECBE90(__int64 a1, unsigned int a2)
{
  __int64 v2; // r9
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // r8
  unsigned int i; // eax
  __int64 v9; // rdx
  unsigned int *v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 result; // rax
  __int64 v15; // rdx

  v2 = 48LL * a2;
  v3 = v2 + *(_QWORD *)(*(_QWORD *)a1 + 208LL);
  v4 = *(_QWORD *)v3;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 160LL) + 88LL * *(unsigned int *)(v3 + 20);
  v6 = *(_DWORD *)(v5 + 24);
  if ( *(_DWORD *)(v3 + 20) == *(_DWORD *)(v3 + 24) )
  {
    *(_DWORD *)(v5 + 24) = *(_DWORD *)(v4 + 16) + v6;
    v7 = *(_QWORD *)(v4 + 32);
  }
  else
  {
    *(_DWORD *)(v5 + 24) = *(_DWORD *)(v4 + 20) + v6;
    v7 = *(_QWORD *)(v4 + 24);
  }
  for ( i = 0; *(_DWORD *)(v5 + 20) > i; *(_DWORD *)(*(_QWORD *)(v5 + 32) + 4 * v9) += *(unsigned __int8 *)(v7 + v9) )
    v9 = i++;
  v10 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)a1 + 208LL) + v2);
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 160LL) + 88LL * v10[6];
  v12 = *(_QWORD *)v10;
  *(_DWORD *)(v11 + 24) += *(_DWORD *)(*(_QWORD *)v10 + 16LL);
  v13 = *(_QWORD *)(v12 + 32);
  result = *(unsigned int *)(v11 + 20);
  if ( (_DWORD)result )
  {
    LODWORD(result) = 0;
    do
    {
      v15 = (unsigned int)result;
      result = (unsigned int)(result + 1);
      *(_DWORD *)(*(_QWORD *)(v11 + 32) + 4 * v15) += *(unsigned __int8 *)(v13 + v15);
    }
    while ( *(_DWORD *)(v11 + 20) > (unsigned int)result );
  }
  return result;
}
