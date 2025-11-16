// Function: sub_148BC60
// Address: 0x148bc60
//
__int64 __fastcall sub_148BC60(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_1479370(a1, a2, a3, a4, a5, a6)
    || (unsigned __int8)sub_148B860(a1, a2, a3, a4, a5, a6)
    || (unsigned __int8)sub_1489CC0(a1, a2, a3, a4, a5, a6) )
  {
    return 1;
  }
  v12 = sub_1480810(a1, a5);
  v10 = sub_1480810(a1, a6);
  return sub_1489CC0(a1, a2, a3, a4, v10, v12);
}
