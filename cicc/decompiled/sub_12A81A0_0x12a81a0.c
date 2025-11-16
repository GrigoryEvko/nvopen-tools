// Function: sub_12A81A0
// Address: 0x12a81a0
//
__int64 __fastcall sub_12A81A0(_QWORD **a1, __int64 *a2, char a3, __int64 a4)
{
  _QWORD *v5; // r12
  __int64 v7; // rdi
  _QWORD *v8; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // [rsp+8h] [rbp-68h] BYREF
  char v20[16]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v21; // [rsp+20h] [rbp-50h]
  char v22[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v23; // [rsp+40h] [rbp-30h]

  v5 = a2;
  v7 = *a2;
  if ( a3 )
  {
    v8 = *a1;
    v21 = 257;
    if ( a4 == v7 )
      return (__int64)v5;
    if ( *((_BYTE *)a2 + 16) <= 0x10u )
      return sub_15A46C0(37, a2, a4, 0);
    v10 = a4;
    v23 = 257;
    v11 = 37;
LABEL_12:
    v12 = sub_15FDBD0(v11, a2, v10, v22, 0);
    v13 = v8[7];
    v5 = (_QWORD *)v12;
    if ( v13 )
    {
      v14 = (unsigned __int64 *)v8[8];
      sub_157E9D0(v13 + 40, v12);
      v15 = v5[3];
      v16 = *v14;
      v5[4] = v14;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      v5[3] = v16 | v15 & 7;
      *(_QWORD *)(v16 + 8) = v5 + 3;
      *v14 = *v14 & 7 | (unsigned __int64)(v5 + 3);
    }
    sub_164B780(v5, v20);
    v17 = v8[6];
    if ( v17 )
    {
      v19 = v8[6];
      sub_1623A60(&v19, v17, 2);
      if ( v5[6] )
        sub_161E7C0(v5 + 6);
      v18 = v19;
      v5[6] = v19;
      if ( v18 )
        sub_1623210(&v19, v18, v5 + 6);
    }
    return (__int64)v5;
  }
  if ( !(unsigned __int8)sub_16430A0(v7, a4) )
    sub_127B550("unexpected: cannot convert return value to return type!", (_DWORD *)(*a1[1] + 36LL), 1);
  v8 = *a1;
  v21 = 257;
  if ( a4 == *a2 )
    return (__int64)v5;
  if ( *((_BYTE *)a2 + 16) > 0x10u )
  {
    v10 = a4;
    v23 = 257;
    v11 = 47;
    goto LABEL_12;
  }
  return sub_15A46C0(47, a2, a4, 0);
}
