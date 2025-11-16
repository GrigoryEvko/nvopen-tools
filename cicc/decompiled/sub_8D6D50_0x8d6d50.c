// Function: sub_8D6D50
// Address: 0x8d6d50
//
__int64 __fastcall sub_8D6D50(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = 0;
  if ( (unsigned int)sub_72EA80(a1, v4, 1) )
  {
    v1 = *(_QWORD *)(v4[0] + 120LL);
    if ( sub_8D3410(v1) )
    {
      v3 = sub_8D46C0(*(_QWORD *)(a1 + 128));
      if ( !sub_8D3410(v3) )
        return sub_8D40F0(v1);
    }
  }
  return v1;
}
