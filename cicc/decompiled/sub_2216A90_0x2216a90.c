// Function: sub_2216A90
// Address: 0x2216a90
//
__int64 __fastcall sub_2216A90(__int64 *a1, int a2)
{
  __int64 v2; // rcx
  __int64 v3; // rbp
  __int64 result; // rax

  v2 = *(_QWORD *)(*a1 - 24);
  v3 = v2 + 1;
  if ( (unsigned __int64)(v2 + 1) > *(_QWORD *)(*a1 - 16) || *(int *)(*a1 - 8) > 0 )
    sub_2216730(a1, v2 + 1);
  result = *a1;
  *(_DWORD *)(result + 4LL * *(_QWORD *)(*a1 - 24)) = a2;
  if ( (_UNKNOWN *)(result - 24) != &unk_4FD67E0 )
  {
    *(_DWORD *)(result - 8) = 0;
    *(_QWORD *)(result - 24) = v3;
    *(_DWORD *)(result + 4 * v3) = 0;
  }
  return result;
}
