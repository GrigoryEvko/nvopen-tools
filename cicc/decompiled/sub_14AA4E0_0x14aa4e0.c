// Function: sub_14AA4E0
// Address: 0x14aa4e0
//
__int64 __fastcall sub_14AA4E0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
  {
    sub_16A4EF0(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = a2;
    return sub_16A4EF0(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = 0;
  }
  return result;
}
