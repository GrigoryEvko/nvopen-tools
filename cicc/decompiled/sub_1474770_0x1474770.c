// Function: sub_1474770
// Address: 0x1474770
//
__int64 __fastcall sub_1474770(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v8; // rax

  v6 = *(_QWORD *)(a3 + 48);
  if ( !(unsigned __int8)sub_148B410(a1, v6, a2, **(_QWORD **)(a3 + 32), a4) )
    return 0;
  v8 = sub_1488A90(a3, a1);
  return sub_1474350(a1, v6, a2, v8, a4);
}
