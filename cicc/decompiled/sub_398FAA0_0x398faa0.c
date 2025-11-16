// Function: sub_398FAA0
// Address: 0x398faa0
//
void __fastcall sub_398FAA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // rax
  _BYTE *v6; // rsi
  __int64 *v7; // [rsp+8h] [rbp-38h] BYREF
  _QWORD *v8; // [rsp+18h] [rbp-28h] BYREF

  v7 = (__int64 *)a2;
  v4 = *sub_398F1A0(
          a1 + 104,
          (unsigned __int8 *)(a2 + 24),
          *(_QWORD *)a2,
          &v7,
          (__int64 (__fastcall **)(__int64, __int64))(a1 + 136));
  v5 = (_QWORD *)sub_145CBF0((__int64 *)a1, 16, 16);
  v5[1] = a3;
  v8 = v5;
  *v5 = &unk_4A3F960;
  v6 = *(_BYTE **)(v4 + 32);
  if ( v6 == *(_BYTE **)(v4 + 40) )
  {
    sub_398F910(v4 + 24, v6, &v8);
  }
  else
  {
    if ( v6 )
    {
      *(_QWORD *)v6 = v5;
      v6 = *(_BYTE **)(v4 + 32);
    }
    *(_QWORD *)(v4 + 32) = v6 + 8;
  }
}
