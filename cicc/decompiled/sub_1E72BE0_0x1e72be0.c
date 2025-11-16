// Function: sub_1E72BE0
// Address: 0x1e72be0
//
__int64 __fastcall sub_1E72BE0(__int64 a1, unsigned int a2, int a3)
{
  __int64 result; // rax

  result = *(unsigned int *)(*(_QWORD *)(a1 + 288) + 4LL * a2);
  if ( (_DWORD)result == -1 )
    return 0;
  if ( *(_DWORD *)(a1 + 24) != 1 )
    return (unsigned int)(a3 + result);
  return result;
}
