// Function: sub_5D3700
// Address: 0x5d3700
//
double __fastcall sub_5D3700(__int64 a1)
{
  __int64 i; // rax
  __int64 v2; // rdx
  _QWORD v4[2]; // [rsp+0h] [rbp-10h] BYREF

  for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4[0] = sub_709B30(*(unsigned __int8 *)(i + 160), a1 + 176);
  v4[1] = v2;
  return COERCE_DOUBLE(sub_12F99A0(v4));
}
