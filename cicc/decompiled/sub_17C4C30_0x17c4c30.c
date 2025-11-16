// Function: sub_17C4C30
// Address: 0x17c4c30
//
__int64 __fastcall sub_17C4C30(_QWORD **a1, __int64 a2, char a3)
{
  _QWORD *v4; // r12
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  char v8; // dl
  __int64 v10; // rax
  _QWORD v11[12]; // [rsp+0h] [rbp-60h] BYREF

  v4 = *a1;
  v5 = (__int64 *)sub_1643270(*a1);
  v11[0] = sub_1643360(v4);
  v11[1] = sub_16471D0(v4, 0);
  v11[2] = sub_1643350(v4);
  if ( a3 )
  {
    v11[3] = sub_1643360(v4);
    v11[4] = sub_1643360(v4);
    v11[5] = sub_1643360(v4);
    v10 = sub_1644EA0(v5, v11, 6, 0);
    v7 = sub_1632190((__int64)a1, (__int64)"__llvm_profile_instrument_range", 31, v10);
  }
  else
  {
    v6 = sub_1644EA0(v5, v11, 3, 0);
    v7 = sub_1632190((__int64)a1, (__int64)"__llvm_profile_instrument_target", 32, v6);
  }
  if ( !*(_BYTE *)(v7 + 16) )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 144LL) )
    {
      v8 = 58;
    }
    else
    {
      v8 = 40;
      if ( !*(_BYTE *)(*(_QWORD *)a2 + 146LL) )
        return v7;
    }
    sub_15E0DF0(v7, 2, v8);
  }
  return v7;
}
