// Function: sub_259AF20
// Address: 0x259af20
//
__int64 __fastcall sub_259AF20(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r13
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  char v9; // [rsp+Fh] [rbp-41h] BYREF
  _QWORD v10[8]; // [rsp+10h] [rbp-40h] BYREF

  v10[0] = a2;
  v10[1] = a1;
  v9 = 0;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_259B800,
                          (__int64)v10,
                          a1,
                          1u,
                          &v9) )
  {
    if ( !v9 )
      *(_BYTE *)(a1 + 96) = *(_BYTE *)(a1 + 97);
    return 1;
  }
  v3 = (_QWORD *)sub_25289A0(a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), a1, 0, 0, 1);
  if ( !v3 )
    return 1;
  v4 = sub_25096F0((_QWORD *)(a1 + 72));
  v5 = sub_25096F0(v3 + 9);
  v6 = v5;
  if ( v5 )
  {
    if ( !sub_B2FC80(v5) )
    {
      v7 = *(_QWORD *)(v6 + 80);
      if ( !v7 )
        BUG();
      v8 = *(_QWORD *)(v7 + 32);
      if ( v8 )
        v8 -= 24;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(*v3 + 112LL))(
              v3,
              a2,
              v8,
              v4,
              0) )
        return 1;
    }
  }
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
