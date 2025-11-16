// Function: sub_1687600
// Address: 0x1687600
//
__int64 __fastcall sub_1687600(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 result; // rax

  if ( !a1 )
    return 1;
  v2 = *a1;
  if ( !v2 )
    return 1;
  result = 0;
  if ( *(_DWORD *)(v2 + 8) >= *(_DWORD *)(*(_QWORD *)v2 + 80LL) )
  {
    sub_16856A0((_QWORD *)v2);
    *a1 = 0;
    return 1;
  }
  return result;
}
