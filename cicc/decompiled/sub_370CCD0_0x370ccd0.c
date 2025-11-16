// Function: sub_370CCD0
// Address: 0x370ccd0
//
unsigned __int64 *__fastcall sub_370CCD0(unsigned __int64 *a1, unsigned int a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  _BYTE v6[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v7; // [rsp+20h] [rbp-30h]

  v2 = sub_37F93E0();
  v7 = 257;
  v3 = sub_22077B0(0x40u);
  v4 = v3;
  if ( v3 )
  {
    sub_C63E60(v3, a2, v2, (__int64)v6);
    *(_QWORD *)v4 = &unk_4A3C5B0;
  }
  *a1 = v4 | 1;
  return a1;
}
