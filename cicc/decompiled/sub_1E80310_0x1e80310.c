// Function: sub_1E80310
// Address: 0x1e80310
//
__int64 __fastcall sub_1E80310(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(a2 + 48);
  if ( *(_DWORD *)(result + 24) == -1 )
    return 0;
  return result;
}
