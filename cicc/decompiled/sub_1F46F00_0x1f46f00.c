// Function: sub_1F46F00
// Address: 0x1f46f00
//
__int64 __fastcall sub_1F46F00(__int64 a1, void *a2, char a3, char a4, unsigned __int8 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  char v10; // dl
  _QWORD *v11; // rsi
  __int64 v12; // r14

  v8 = sub_1F462F0(a1, (__int64)a2);
  v9 = sub_1F446D0(a2, v8);
  if ( !v9 )
    return 0;
  v11 = (_QWORD *)v9;
  if ( !v10 )
    v11 = (_QWORD *)sub_16369A0(v9, v9);
  v12 = v11[2];
  sub_1F46490(a1, v11, a3, a4, a5);
  return v12;
}
