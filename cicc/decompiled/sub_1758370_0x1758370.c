// Function: sub_1758370
// Address: 0x1758370
//
__int64 __fastcall sub_1758370(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD **v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v13; // [rsp-10h] [rbp-50h]
  _QWORD *v14; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD ***)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
  {
    v14 = v9[4];
    v10 = (__int64 *)sub_1643320(*v9);
    v11 = (__int64)sub_16463B0(v10, (unsigned int)v14);
  }
  else
  {
    v11 = sub_1643320(*v9);
  }
  sub_15FEC10(a1, v11, 52, a2, a3, a4, a5, 0);
  return v13;
}
