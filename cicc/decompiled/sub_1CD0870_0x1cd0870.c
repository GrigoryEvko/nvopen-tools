// Function: sub_1CD0870
// Address: 0x1cd0870
//
__int64 __fastcall sub_1CD0870(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD **v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v14; // [rsp-10h] [rbp-50h]
  _QWORD *v15; // [rsp+8h] [rbp-38h]

  v10 = *(_QWORD ***)a4;
  if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 16 )
  {
    v15 = v10[4];
    v11 = (__int64 *)sub_1643320(*v10);
    v12 = (__int64)sub_16463B0(v11, (unsigned int)v15);
  }
  else
  {
    v12 = sub_1643320(*v10);
  }
  sub_15FEC10(a1, v12, 51, a3, a4, a5, a6, a2);
  return v14;
}
