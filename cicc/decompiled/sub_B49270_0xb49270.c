// Function: sub_B49270
// Address: 0xb49270
//
__int64 __fastcall sub_B49270(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_A74660((_QWORD *)(a1 + 72));
  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    return v1;
  v4[0] = *(_QWORD *)(v2 + 120);
  return (unsigned int)sub_A74660(v4) | v1;
}
