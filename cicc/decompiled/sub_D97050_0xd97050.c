// Function: sub_D97050
// Address: 0xd97050
//
__int64 __fastcall sub_D97050(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rdx
  _QWORD v5[2]; // [rsp+0h] [rbp-10h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(a2 + 8) == 14 )
    return (unsigned int)sub_AE43F0(v2, a2);
  v5[0] = sub_9208B0(v2, a2);
  v5[1] = v3;
  return sub_CA1930(v5);
}
