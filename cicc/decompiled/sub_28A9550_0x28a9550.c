// Function: sub_28A9550
// Address: 0x28a9550
//
bool __fastcall sub_28A9550(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char v5; // r8
  bool result; // al
  unsigned __int8 *v7; // rax
  __int64 v8; // r12
  __int64 i; // rbx
  unsigned __int8 *v10; // rdi
  bool v11[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = sub_B43CB0(a2);
  v5 = sub_B2D610(v4, 41);
  result = 0;
  if ( !v5 )
  {
    v7 = sub_98ACB0(a1, 6u);
    if ( !(unsigned __int8)sub_CF7590(v7, v11) || (result = v11[0]) )
    {
      v8 = a3 + 24;
      for ( i = a2 + 24; v8 != i; i = *(_QWORD *)(i + 8) )
      {
        v10 = (unsigned __int8 *)(i - 24);
        if ( !i )
          v10 = 0;
        if ( (unsigned __int8)sub_B46790(v10, 0) )
          break;
      }
      return v8 != i;
    }
  }
  return result;
}
