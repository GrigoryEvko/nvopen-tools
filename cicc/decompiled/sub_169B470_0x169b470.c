// Function: sub_169B470
// Address: 0x169b470
//
__int64 __fastcall sub_169B470(_BYTE *a1)
{
  int v2; // r12d
  __int64 v3; // rax
  unsigned int v4; // r8d

  if ( (a1[18] & 7) != 1 )
    return 0;
  v2 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v3 = sub_16984A0((__int64)a1);
  LOBYTE(v4) = (unsigned int)sub_16A70B0(v3, (unsigned int)(v2 - 2)) == 0;
  return v4;
}
