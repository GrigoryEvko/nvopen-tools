// Function: sub_15C70F0
// Address: 0x15c70f0
//
__int64 __fastcall sub_15C70F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8

  v1 = sub_15C70A0(a1);
  v2 = 0;
  if ( *(_DWORD *)(v1 + 8) == 2 )
    return *(_QWORD *)(v1 - 8);
  return v2;
}
