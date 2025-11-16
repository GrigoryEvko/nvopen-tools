// Function: sub_34184D0
// Address: 0x34184d0
//
__int64 __fastcall sub_34184D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 200LL);
  v2 = *(_QWORD *)(v1 + 96);
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 <= 0x40 )
  {
    LOBYTE(v1) = *(_QWORD *)(v2 + 24) == 1;
  }
  else
  {
    LODWORD(v1) = sub_C444A0(v2 + 24);
    LOBYTE(v1) = v3 - 1 == (_DWORD)v1;
  }
  return (unsigned int)v1 ^ 1;
}
