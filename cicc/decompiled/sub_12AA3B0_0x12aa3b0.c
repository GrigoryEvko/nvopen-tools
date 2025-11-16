// Function: sub_12AA3B0
// Address: 0x12aa3b0
//
__int64 __fastcall sub_12AA3B0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // r12
  unsigned __int64 *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  char v17[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  if ( a4 == *(_QWORD *)a3 )
    return a3;
  if ( *(_BYTE *)(a3 + 16) <= 0x10u )
    return sub_15A46C0(a2, a3, a4, 0);
  v18 = 257;
  v8 = sub_15FDBD0(a2, a3, a4, v17, 0);
  v9 = a1[1];
  v10 = (_QWORD *)v8;
  if ( v9 )
  {
    v11 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v9 + 40, v8);
    v12 = v10[3];
    v13 = *v11;
    v10[4] = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v10 + 3;
    *v11 = *v11 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, a5);
  v14 = *a1;
  if ( *a1 )
  {
    v16 = *a1;
    sub_1623A60(&v16, v14, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v15 = v16;
    v10[6] = v16;
    if ( v15 )
      sub_1623210(&v16, v15, v10 + 6);
  }
  return (__int64)v10;
}
