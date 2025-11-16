// Function: sub_A6A1D0
// Address: 0xa6a1d0
//
__int64 __fastcall sub_A6A1D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  sub_A6A190(a1);
  v4 = sub_C92610(a2, a3);
  result = sub_C92860(a1 + 264, a2, a3, v4);
  if ( (_DWORD)result != -1 )
  {
    v6 = *(_QWORD *)(a1 + 264);
    v7 = v6 + 8LL * (int)result;
    if ( v7 == v6 + 8LL * *(unsigned int *)(a1 + 272) )
      return 0xFFFFFFFFLL;
    else
      return *(unsigned int *)(*(_QWORD *)v7 + 8LL);
  }
  return result;
}
