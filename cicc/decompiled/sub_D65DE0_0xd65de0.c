// Function: sub_D65DE0
// Address: 0xd65de0
//
__int64 __fastcall sub_D65DE0(__int64 a1, __int64 a2, __int64 a3)
{
  if ( (unsigned __int8)sub_B2F6B0(a3) )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  else
  {
    sub_D62600(a1, a2, *(_QWORD *)(a3 - 32));
    return a1;
  }
}
