// Function: sub_E807A0
// Address: 0xe807a0
//
void __fastcall sub_E807A0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  a3 = (unsigned __int16)a3;
  if ( *(_BYTE *)(a4 + 18) )
    a3 = (unsigned __int16)a3 | 0x10000;
  *(_WORD *)(a1 + 1) = a3;
  *(_BYTE *)a1 = 2;
  *(_BYTE *)(a1 + 3) = BYTE2(a3);
  *(_QWORD *)(a1 + 8) = a5;
  *(_QWORD *)(a1 + 16) = a2;
}
