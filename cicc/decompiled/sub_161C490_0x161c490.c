// Function: sub_161C490
// Address: 0x161c490
//
__int64 __fastcall sub_161C490(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  _QWORD *v15; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+18h] [rbp-58h]
  _QWORD v17[10]; // [rsp+20h] [rbp-50h] BYREF

  v6 = sub_1627350(*a1, 0, 0, 2, 1);
  v15 = v17;
  v7 = v6;
  v16 = 0x300000001LL;
  v17[0] = v6;
  if ( a4 )
  {
    v17[1] = a4;
    LODWORD(v16) = 2;
  }
  if ( a3 )
  {
    v15[(unsigned int)v16] = sub_161BD10(a1, a2, a3);
    v9 = v15;
    v8 = (unsigned int)(v16 + 1);
    LODWORD(v16) = v16 + 1;
  }
  else
  {
    v8 = (unsigned int)v16;
    v9 = v17;
  }
  v10 = sub_1627350(*a1, v9, v8, 0, 1);
  sub_1630830(v10, 0, v10);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  if ( v7 )
    sub_16307F0(v7, 0, v11, v12, v13);
  return v10;
}
