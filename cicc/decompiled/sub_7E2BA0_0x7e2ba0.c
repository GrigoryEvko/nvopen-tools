// Function: sub_7E2BA0
// Address: 0x7e2ba0
//
__int64 __fastcall sub_7E2BA0(__int64 a1)
{
  __int64 result; // rax

  if ( *(_DWORD *)a1 )
  {
    result = (unsigned int)(*(_DWORD *)a1 - 1);
    if ( (unsigned int)result > 1 )
      sub_721090();
    *(_BYTE *)(a1 + 24) = 1;
  }
  else
  {
    result = *(_QWORD *)(a1 + 8);
    *(_BYTE *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = result;
  }
  return result;
}
