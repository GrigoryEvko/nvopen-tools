// Function: sub_A18590
// Address: 0xa18590
//
__int64 __fastcall sub_A18590(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v6; // r13d
  unsigned int v7; // r8d
  __int64 v8; // rax
  __int64 result; // rax
  int v10; // eax

  v6 = sub_A3F3B0(a1 + 24);
  v7 = a3 - v6;
  v8 = *(unsigned int *)(a4 + 8);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, a4 + 16, v8 + 1, 4);
    v8 = *(unsigned int *)(a4 + 8);
    v7 = a3 - v6;
  }
  *(_DWORD *)(*(_QWORD *)a4 + 4 * v8) = v7;
  result = 0;
  ++*(_DWORD *)(a4 + 8);
  if ( v6 >= a3 )
  {
    v10 = sub_A172F0(a1 + 24, *(_QWORD *)(a2 + 8));
    sub_9C8C60(a4, v10);
    return 1;
  }
  return result;
}
