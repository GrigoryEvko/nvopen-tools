// Function: sub_242DF10
// Address: 0x242df10
//
__int64 __fastcall sub_242DF10(__int64 a1)
{
  __int64 v1; // rbp
  _QWORD v3[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v4; // [rsp-18h] [rbp-18h]
  __int64 v5; // [rsp-8h] [rbp-8h]

  *(_BYTE *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_WORD *)(a1 + 6) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 56) = 0;
  *(_WORD *)a1 = 257;
  *(_BYTE *)(a1 + 7) = qword_4FE3B48;
  if ( qword_4FE3C30 != 4 )
  {
    v5 = v1;
    v3[0] = "Invalid -default-gcov-version: ";
    v3[2] = &qword_4FE3C28;
    v4 = 1027;
    sub_C64D30((__int64)v3, 0);
  }
  *(_DWORD *)(a1 + 2) = *(_DWORD *)qword_4FE3C28;
  return a1;
}
