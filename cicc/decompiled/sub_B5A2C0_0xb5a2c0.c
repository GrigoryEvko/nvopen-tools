// Function: sub_B5A2C0
// Address: 0xb5a2c0
//
__int64 __fastcall sub_B5A2C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v4; // [rsp+8h] [rbp-18h]

  v1 = sub_B5A250(a1);
  if ( v1 )
    v2 = *(_QWORD *)(v1 + 8);
  else
    v2 = *(_QWORD *)(a1 + 8);
  BYTE4(v4) = *(_BYTE *)(v2 + 8) == 18;
  LODWORD(v4) = *(_DWORD *)(v2 + 32);
  return v4;
}
