// Function: sub_86A1D0
// Address: 0x86a1d0
//
_BYTE *__fastcall sub_86A1D0(__int64 a1, char a2, __int64 a3)
{
  _BYTE *v4; // r12
  int v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_7296C0(v6);
  v4 = sub_727110();
  sub_729730(v6[0]);
  *((_QWORD *)v4 + 3) = a1;
  v4[16] = a2;
  *((_QWORD *)v4 + 4) = a3;
  return v4;
}
