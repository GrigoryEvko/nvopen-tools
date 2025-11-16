// Function: sub_B543C0
// Address: 0xb543c0
//
__int64 __fastcall sub_B543C0(__int64 a1, unsigned int a2)
{
  __int64 v3; // [rsp+0h] [rbp-8h]

  if ( *(_BYTE *)(a1 + 56) )
  {
    BYTE4(v3) = 1;
    LODWORD(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL * a2);
  }
  else
  {
    BYTE4(v3) = 0;
  }
  return v3;
}
