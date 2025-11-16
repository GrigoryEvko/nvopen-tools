// Function: sub_12F51F0
// Address: 0x12f51f0
//
__int64 __fastcall sub_12F51F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  int v7; // eax
  unsigned int v8; // r8d
  __int64 v9; // rax

  if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
    return 1;
  v3 = *a1;
  v4 = sub_1649960(a2);
  v5 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  v7 = sub_16D1B30(v3, v4, v6);
  if ( v7 == -1 )
    v9 = *(_QWORD *)v3 + 8LL * *(unsigned int *)(v3 + 8);
  else
    v9 = *(_QWORD *)v3 + 8LL * v7;
  LOBYTE(v8) = v5 == v9;
  return v8;
}
