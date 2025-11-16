// Function: sub_2FF1CA0
// Address: 0x2ff1ca0
//
__int64 __fastcall sub_2FF1CA0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // [rsp+8h] [rbp-88h] BYREF
  __int64 *v8; // [rsp+10h] [rbp-80h] BYREF
  __int64 v9; // [rsp+18h] [rbp-78h]
  __int64 v10; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v11[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v12; // [rsp+40h] [rbp-50h] BYREF
  __int64 v13[2]; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+60h] [rbp-30h] BYREF

  if ( LOBYTE(qword_4F813A8[8]) )
  {
    v3 = (_QWORD *)sub_3581200(2);
    sub_2FF0E80(a1, v3, 0);
    sub_2FEF140((__int64)&v8, *(_QWORD *)(a1 + 256));
    if ( v9 && !(_BYTE)qword_5027F28 )
    {
      v4 = *(_QWORD *)(a1 + 256);
      v7 = 0;
      sub_2FEF0A0((__int64)v11, v4);
      v13[0] = (__int64)v14;
      sub_2FEEBD0(v13, v8, (__int64)v8 + v9);
      v5 = (_QWORD *)sub_3585380(v13, v11, 2, &v7);
      sub_2FF0E80(a1, v5, 0);
      if ( (_QWORD *)v13[0] != v14 )
        j_j___libc_free_0(v13[0]);
      if ( (__int64 *)v11[0] != &v12 )
        j_j___libc_free_0(v11[0]);
      v6 = v7;
      if ( v7 && !_InterlockedSub((volatile signed __int32 *)(v7 + 8), 1u) )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    }
    if ( v8 != &v10 )
      j_j___libc_free_0((unsigned __int64)v8);
  }
  result = sub_2FF12A0(a1, &unk_503BDC4, 0);
  if ( result )
  {
    if ( byte_5029B88 )
      return sub_2FF12A0(a1, &unk_503BDBC, 0);
  }
  return result;
}
