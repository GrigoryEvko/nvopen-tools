// Function: sub_156E470
// Address: 0x156e470
//
__int64 __fastcall sub_156E470(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  unsigned int v9; // r15d
  _QWORD *v10; // r14
  __int64 v11; // rdi
  unsigned __int64 *v12; // r12
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // [rsp+8h] [rbp-58h] BYREF
  char v18[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v19; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A2BF0(a2);
  v19 = 257;
  v8 = sub_15FB5B0(a2, v18, 0);
  v9 = *((_DWORD *)a1 + 10);
  v10 = (_QWORD *)v8;
  if ( a4 || (a4 = a1[4]) != 0 )
    sub_1625C10(v8, 3, a4);
  sub_15F2440(v10, v9);
  v11 = a1[1];
  if ( v11 )
  {
    v12 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, a3);
  v15 = *a1;
  if ( *a1 )
  {
    v17 = *a1;
    sub_1623A60(&v17, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v17;
    v10[6] = v17;
    if ( v16 )
      sub_1623210(&v17, v16, v10 + 6);
  }
  return (__int64)v10;
}
