// Function: sub_1948D70
// Address: 0x1948d70
//
__int64 __fastcall sub_1948D70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx

  v4 = sub_1456040(a1);
  v5 = sub_145CF80(a3, v4, 0, 0);
  if ( sub_146D950(a3, a1, a2) )
    return sub_148B410(a3, a2, 0x27u, a1, v5);
  else
    return 0;
}
