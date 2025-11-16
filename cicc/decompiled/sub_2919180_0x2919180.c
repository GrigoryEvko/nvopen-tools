// Function: sub_2919180
// Address: 0x2919180
//
__int64 __fastcall sub_2919180(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v1 = 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v2 = *(_QWORD *)(a1 + v1);
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 <= 0x40 )
  {
    LOBYTE(v1) = *(_QWORD *)(v2 + 24) == 0;
  }
  else
  {
    LODWORD(v1) = sub_C444A0(v2 + 24);
    LOBYTE(v1) = v3 == (_DWORD)v1;
  }
  return (unsigned int)v1 ^ 1;
}
