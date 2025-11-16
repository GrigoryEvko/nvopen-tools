// Function: sub_19E1860
// Address: 0x19e1860
//
__int64 __fastcall sub_19E1860(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int v6; // eax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 result; // rax
  unsigned __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx

  v6 = 0;
  v8 = a2;
  if ( a2 )
  {
    v8 = a2 - 1LL;
    if ( a2 != 1 )
    {
      _BitScanReverse64(&v8, v8);
      v6 = 64 - (v8 ^ 0x3F);
      v8 = 8LL * v6;
    }
  }
  v9 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v9 > v6 )
    goto LABEL_5;
  v12 = v6 + 1;
  if ( v12 < v9 )
  {
    *(_DWORD *)(a3 + 8) = v12;
    goto LABEL_5;
  }
  if ( v12 <= v9 )
  {
LABEL_5:
    v10 = *(_QWORD *)a3;
    goto LABEL_6;
  }
  if ( v12 > *(unsigned int *)(a3 + 12) )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), v12, 8, a5, a6);
    v9 = *(unsigned int *)(a3 + 8);
  }
  v10 = *(_QWORD *)a3;
  v13 = (_QWORD *)(*(_QWORD *)a3 + 8 * v9);
  v14 = *(_QWORD *)a3 + 8 * v12;
  if ( v13 != (_QWORD *)v14 )
  {
    do
    {
      if ( v13 )
        *v13 = 0;
      ++v13;
    }
    while ( (_QWORD *)v14 != v13 );
    v10 = *(_QWORD *)a3;
  }
  *(_DWORD *)(a3 + 8) = v12;
LABEL_6:
  *a1 = *(_QWORD *)(v10 + v8);
  result = *(_QWORD *)a3;
  *(_QWORD *)(*(_QWORD *)a3 + v8) = a1;
  return result;
}
