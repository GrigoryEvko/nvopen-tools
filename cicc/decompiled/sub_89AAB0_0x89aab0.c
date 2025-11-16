// Function: sub_89AAB0
// Address: 0x89aab0
//
__int64 __fastcall sub_89AAB0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  _BYTE *v6; // r14
  __int64 v7; // rax
  _BYTE *v8; // r13
  __int64 v9; // r12
  __int64 v10; // rax

  if ( !a1 )
    return 0;
  if ( !a2 )
    return 0;
  v5 = ((__int64 (*)(void))sub_8C9880)();
  v6 = sub_89A800(v5);
  v7 = sub_8C9880(a2);
  v8 = sub_89A800(v7);
  v9 = sub_8794A0(v6);
  v10 = sub_8794A0(v8);
  return sub_89B9E0(v9, v10, a3, 0);
}
