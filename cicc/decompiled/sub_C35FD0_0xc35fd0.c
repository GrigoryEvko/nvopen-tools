// Function: sub_C35FD0
// Address: 0xc35fd0
//
__int64 __fastcall sub_C35FD0(_BYTE *a1)
{
  int v2; // r12d
  __int64 v3; // rax
  unsigned int v4; // r8d

  if ( (a1[20] & 7) != 1 || (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) <= 1 )
    return 0;
  v2 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v3 = sub_C33930((__int64)a1);
  LOBYTE(v4) = (unsigned int)sub_C45D90(v3, (unsigned int)(v2 - 2)) == 0;
  return v4;
}
