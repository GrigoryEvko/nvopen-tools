// Function: sub_1D172F0
// Address: 0x1d172f0
//
__int64 __fastcall sub_1D172F0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 *v7; // rax
  _QWORD v8[4]; // [rsp+0h] [rbp-20h] BYREF

  v8[0] = a2;
  v3 = *(_QWORD **)(a1 + 48);
  v8[1] = a3;
  if ( (_BYTE)a2 == 0xFE )
  {
    v7 = (__int64 *)sub_1643330(v3);
    v4 = sub_1646BA0(v7, 0);
  }
  else
  {
    v4 = sub_1F58E60(v8, v3);
  }
  v5 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
  return sub_15A9FE0(v5, v4);
}
