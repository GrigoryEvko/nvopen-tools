// Function: sub_70D5B0
// Address: 0x70d5b0
//
__int64 __fastcall sub_70D5B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  _QWORD v5[6]; // [rsp+0h] [rbp-30h] BYREF

  v5[0] = 0;
  v5[1] = 0;
  v2 = sub_7A30C0(a1, 0, 0, a2);
  if ( v2 && (unsigned int)sub_8D2FB0(*(_QWORD *)(a2 + 128)) )
  {
    v4 = sub_8D46C0(*(_QWORD *)(a2 + 128));
    *(_QWORD *)(a2 + 128) = sub_72D2E0(v4, 0);
  }
  sub_67E3D0(v5);
  return v2;
}
