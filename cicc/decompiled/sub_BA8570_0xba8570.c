// Function: sub_BA8570
// Address: 0xba8570
//
void __fastcall sub_BA8570(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax

  *(_QWORD *)(a2 + 40) = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 96);
    if ( v2 )
    {
      v3 = sub_BD5C70(a2);
      sub_BD8AE0(v2, v3);
    }
  }
}
