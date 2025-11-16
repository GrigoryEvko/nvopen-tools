// Function: sub_CB7080
// Address: 0xcb7080
//
__int64 __fastcall sub_CB7080(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  *(_BYTE *)(a1 + 52) = 0;
  if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a1 + 16) )
    sub_CB5AE0((__int64 *)a1);
  result = sub_C86220(*(_DWORD *)(a1 + 48), a2);
  if ( (_DWORD)result )
  {
    *(_DWORD *)(a1 + 72) = result;
    *(_QWORD *)(a1 + 80) = v3;
  }
  *(_DWORD *)(a1 + 48) = -1;
  return result;
}
