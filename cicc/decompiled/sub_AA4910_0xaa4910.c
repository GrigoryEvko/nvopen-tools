// Function: sub_AA4910
// Address: 0xaa4910
//
void __fastcall sub_AA4910(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax

  *(_QWORD *)(a2 + 40) = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v2 = sub_AA4890(a1 - 48);
    if ( v2 )
    {
      v3 = sub_BD5C70(a2);
      sub_BD8AE0(v2, v3);
    }
  }
}
