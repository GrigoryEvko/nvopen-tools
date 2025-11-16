// Function: sub_B445A0
// Address: 0xb445a0
//
bool __fastcall sub_B445A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = *(_QWORD *)(a1 + 40);
  if ( *(char *)(v3 + 3) >= 0 )
    sub_AA6050(v3);
  return *(_DWORD *)(a1 + 56) < *(_DWORD *)(a2 + 56);
}
