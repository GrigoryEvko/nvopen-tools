// Function: sub_211A870
// Address: 0x211a870
//
__int64 __fastcall sub_211A870(__int64 *a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  unsigned __int8 *v6; // rax
  __int64 v7; // r13
  _BYTE v9[16]; // [rsp+0h] [rbp-50h] BYREF

  v6 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v7 = *v6;
  sub_1F40D10((__int64)v9, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v7, *((_QWORD *)v6 + 1));
  if ( (_BYTE)v7 == v9[8] && (_BYTE)v7 && *(_QWORD *)(*a1 + 8 * v7 + 120) )
    return a2;
  else
    return sub_200D2A0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a4, a5, a6);
}
