// Function: sub_16E8B10
// Address: 0x16e8b10
//
__int64 __fastcall sub_16E8B10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  *(_BYTE *)(a1 + 40) = 0;
  if ( *(_QWORD *)(a1 + 24) != *(_QWORD *)(a1 + 8) )
    sub_16E7BA0((__int64 *)a1);
  result = sub_16C6980(*(_DWORD *)(a1 + 36), a2);
  if ( (_DWORD)result )
  {
    *(_DWORD *)(a1 + 48) = result;
    *(_QWORD *)(a1 + 56) = v3;
  }
  *(_DWORD *)(a1 + 36) = -1;
  return result;
}
