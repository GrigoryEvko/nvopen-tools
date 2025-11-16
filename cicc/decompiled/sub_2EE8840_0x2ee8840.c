// Function: sub_2EE8840
// Address: 0x2ee8840
//
__int64 __fastcall sub_2EE8840(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(a2 + 24);
  if ( *(_DWORD *)(result + 28) == -1 )
    return 0;
  return result;
}
