// Function: sub_1545880
// Address: 0x1545880
//
void __fastcall sub_1545880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // rbx

  for ( i = *(_QWORD *)(a2 + 80); a2 + 72 != i; i = *(_QWORD *)(i + 8) )
    sub_1545830(a1, i, a3, a4);
}
