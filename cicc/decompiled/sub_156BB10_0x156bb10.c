// Function: sub_156BB10
// Address: 0x156bb10
//
__int64 __fastcall sub_156BB10(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _BYTE v8[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v9; // [rsp+10h] [rbp-30h]

  if ( a2[16] <= 0x10u && (unsigned __int8)sub_1596070(a2) )
    return a3;
  v6 = sub_156A930(a1, a2, *(_QWORD *)(*(_QWORD *)a3 + 32LL));
  v9 = 257;
  return sub_156B790(a1, v6, a3, a4, (__int64)v8, 0);
}
