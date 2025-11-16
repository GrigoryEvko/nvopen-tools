// Function: sub_CE9220
// Address: 0xce9220
//
__int64 __fastcall sub_CE9220(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned int v3; // eax
  _DWORD v4[5]; // [rsp+Ch] [rbp-14h] BYREF

  if ( ((*(_WORD *)(a1 + 2) >> 4) & 0x3FF) == 0x47 )
    return 1;
  v1 = sub_B2D620(a1, "nvvm.kernel", 0xBu);
  if ( (_BYTE)v1 )
    return 1;
  if ( (unsigned __int8)sub_B2D620(a1, "nvvm.annotations_transplanted", 0x1Du) )
    return v1;
  v4[0] = 0;
  v3 = sub_CE7ED0(a1, "kernel", 6u, v4);
  if ( (_BYTE)v3 )
    return v3;
  return v1;
}
