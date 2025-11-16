// Function: sub_15E7E90
// Address: 0x15e7e90
//
_QWORD *__fastcall sub_15E7E90(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v10; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  v5 = sub_15E7150(a1, a2);
  if ( !a3 )
  {
    v10 = sub_1643360(a1[3]);
    a3 = sub_159C470(v10, -1, 0);
  }
  v6 = a1[1];
  v11[0] = a3;
  v11[1] = v5;
  v7 = *(__int64 **)(*(_QWORD *)(v6 + 56) + 40LL);
  v12[0] = *v5;
  v8 = sub_15E26F0(v7, 116, v12, 1);
  v13 = 257;
  return sub_15E6DE0(v8, (int)v11, 2, a1, (int)v12, 0, 0, 0);
}
