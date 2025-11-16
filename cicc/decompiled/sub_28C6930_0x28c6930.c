// Function: sub_28C6930
// Address: 0x28c6930
//
__int64 __fastcall sub_28C6930(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // eax
  unsigned int v8; // r13d

  *a1 = a3;
  a1[2] = a4;
  a1[3] = a5;
  a1[4] = a6;
  a1[5] = a7;
  a1[1] = sub_B2BEC0(a2);
  v7 = 0;
  do
  {
    v8 = v7;
    v7 = sub_28C5020((__int64)a1);
  }
  while ( (_BYTE)v7 );
  return v8;
}
