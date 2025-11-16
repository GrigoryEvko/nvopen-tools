// Function: sub_156E040
// Address: 0x156e040
//
_QWORD *__fastcall sub_156E040(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  _QWORD *v9; // r12
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // [rsp+0h] [rbp-60h]
  unsigned int v20; // [rsp+8h] [rbp-58h]
  _QWORD v21[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v22; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v19 = a3, v8 = sub_15A2A30(19, a2, a3, 0, 0), a3 = v19, (v9 = (_QWORD *)v8) == 0) )
  {
    v22 = 257;
    v11 = sub_15FB440(19, a2, a3, v21, 0);
    v12 = *((_DWORD *)a1 + 10);
    v9 = (_QWORD *)v11;
    if ( a5 || (a5 = a1[4]) != 0 )
    {
      v20 = *((_DWORD *)a1 + 10);
      sub_1625C10(v11, 3, a5);
      v12 = v20;
    }
    sub_15F2440(v9, v12);
    v13 = a1[1];
    if ( v13 )
    {
      v14 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v13 + 40, v9);
      v15 = v9[3];
      v16 = *v14;
      v9[4] = v14;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v16 | v15 & 7;
      *(_QWORD *)(v16 + 8) = v9 + 3;
      *v14 = *v14 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780(v9, a4);
    v17 = *a1;
    if ( *a1 )
    {
      v21[0] = *a1;
      sub_1623A60(v21, v17, 2);
      if ( v9[6] )
        sub_161E7C0(v9 + 6);
      v18 = v21[0];
      v9[6] = v21[0];
      if ( v18 )
        sub_1623210(v21, v18, v9 + 6);
    }
  }
  return v9;
}
