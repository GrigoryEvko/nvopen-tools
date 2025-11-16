// Function: sub_2E4FCA0
// Address: 0x2e4fca0
//
__int64 __fastcall sub_2E4FCA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // eax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  int v13; // ecx
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // r8
  int v16; // eax
  unsigned __int64 v17; // r13
  int v18; // eax

  v7 = *(_DWORD *)(a2 + 64);
  if ( *(_DWORD *)(a1 + 64) >= v7 )
    goto LABEL_2;
  v13 = *(_DWORD *)(a1 + 64) & 0x3F;
  if ( v13 )
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v13);
  v14 = *(unsigned int *)(a1 + 8);
  *(_DWORD *)(a1 + 64) = v7;
  v15 = (v7 + 63) >> 6;
  if ( v15 == v14 )
    goto LABEL_10;
  if ( v15 < v14 )
  {
    *(_DWORD *)(a1 + 8) = (v7 + 63) >> 6;
LABEL_10:
    v16 = v7 & 0x3F;
    if ( !v16 )
      goto LABEL_2;
    goto LABEL_11;
  }
  v17 = v15 - v14;
  if ( v15 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 8u, v15, a6);
    v14 = *(unsigned int *)(a1 + 8);
  }
  if ( 8 * v17 )
  {
    memset((void *)(*(_QWORD *)a1 + 8 * v14), 0, 8 * v17);
    LODWORD(v14) = *(_DWORD *)(a1 + 8);
  }
  v18 = *(_DWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 8) = v17 + v14;
  v16 = v18 & 0x3F;
  if ( v16 )
LABEL_11:
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v16);
LABEL_2:
  result = 0;
  v9 = *(unsigned int *)(a2 + 8);
  v10 = 8 * v9;
  if ( (_DWORD)v9 )
  {
    do
    {
      v11 = (_QWORD *)(result + *(_QWORD *)a1);
      v12 = *(_QWORD *)(*(_QWORD *)a2 + result);
      result += 8;
      *v11 |= v12;
    }
    while ( result != v10 );
  }
  return result;
}
