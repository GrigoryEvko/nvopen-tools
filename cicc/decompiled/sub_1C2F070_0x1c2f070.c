// Function: sub_1C2F070
// Address: 0x1c2f070
//
__int64 __fastcall sub_1C2F070(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r13d
  _DWORD v4[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( ((*(_WORD *)(a1 + 18) >> 4) & 0x3FF) == 0x47 )
    return 1;
  LOBYTE(v1) = sub_15602E0((_QWORD *)(a1 + 112), "nvvm.kernel", 0xBu);
  v2 = v1;
  if ( (_BYTE)v1 )
    return 1;
  if ( sub_15602E0((_QWORD *)(a1 + 112), "nvvm.annotations_transplanted", 0x1Du) )
    return v2;
  v4[0] = 0;
  return (unsigned int)sub_1C2E690(a1, "kernel", 6u, v4);
}
