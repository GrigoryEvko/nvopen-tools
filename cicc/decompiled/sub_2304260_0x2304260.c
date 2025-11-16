// Function: sub_2304260
// Address: 0x2304260
//
__int64 *__fastcall sub_2304260(__int64 *a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  __int64 v4; // rax
  _BYTE v6[17]; // [rsp+Fh] [rbp-11h] BYREF

  sub_E00540(v6, a2 + 8, a3);
  v3 = v6[0];
  v4 = sub_22077B0(0x10u);
  if ( v4 )
  {
    *(_BYTE *)(v4 + 8) = v3;
    *(_QWORD *)v4 = &unk_4A15AA0;
  }
  *a1 = v4;
  return a1;
}
