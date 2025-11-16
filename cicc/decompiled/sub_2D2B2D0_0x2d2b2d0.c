// Function: sub_2D2B2D0
// Address: 0x2d2b2d0
//
__int64 __fastcall sub_2D2B2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // r8d
  unsigned int i; // edx
  __int64 v10; // rsi
  __int64 v11; // rcx
  int v12; // r8d
  __int64 result; // rax

  v6 = *(_QWORD *)a1;
  v7 = *(unsigned int *)(*(_QWORD *)a1 + 192LL);
  if ( (_DWORD)v7 )
    return sub_2D2B0B0(a1, 1, v7, a4, a5, a6);
  v8 = *(_DWORD *)(v6 + 196);
  for ( i = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4) + 1;
        v8 != i;
        *(_DWORD *)(v6 + 4 * v11 + 128) = *(_DWORD *)(v6 + 4 * v10 + 128) )
  {
    v10 = i;
    v11 = i++ - 1;
    *(_DWORD *)(v6 + 8 * v11) = *(_DWORD *)(v6 + 8 * v10);
    *(_DWORD *)(v6 + 8 * v11 + 4) = *(_DWORD *)(v6 + 8 * v10 + 4);
  }
  v12 = v8 - 1;
  *(_DWORD *)(v6 + 196) = v12;
  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(result + 8) = v12;
  return result;
}
