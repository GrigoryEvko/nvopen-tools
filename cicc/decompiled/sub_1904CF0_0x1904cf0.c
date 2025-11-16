// Function: sub_1904CF0
// Address: 0x1904cf0
//
__int64 __fastcall sub_1904CF0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _QWORD *v5; // r12
  __int64 v7; // r14
  unsigned int v8; // r15d
  unsigned int v9; // eax
  __int64 **v11; // rdx
  int v12; // edi
  int v13; // edi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  unsigned __int8 *v24; // [rsp+18h] [rbp-58h] BYREF
  char v25[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  v5 = (_QWORD *)a2;
  v7 = *(_QWORD *)a2;
  v8 = sub_16431D0(*(_QWORD *)a2);
  v9 = sub_16431D0(a3);
  if ( v8 >= v9 )
  {
    if ( v7 == a3 || v8 == v9 )
      return (__int64)v5;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v11 = (__int64 **)a3;
      v12 = 36;
      return sub_15A46C0(v12, (__int64 ***)a2, v11, 0);
    }
    v14 = a3;
    v26 = 257;
    v13 = 36;
LABEL_12:
    v15 = sub_15FDBD0(v13, a2, v14, (__int64)v25, 0);
    v16 = a1[1];
    v5 = (_QWORD *)v15;
    if ( v16 )
    {
      v17 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v16 + 40, v15);
      v18 = v5[3];
      v19 = *v17;
      v5[4] = v17;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      v5[3] = v19 | v18 & 7;
      *(_QWORD *)(v19 + 8) = v5 + 3;
      *v17 = *v17 & 7 | (unsigned __int64)(v5 + 3);
    }
    sub_164B780((__int64)v5, a4);
    v20 = *a1;
    if ( *a1 )
    {
      v24 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v24, v20, 2);
      v21 = v5[6];
      if ( v21 )
        sub_161E7C0((__int64)(v5 + 6), v21);
      v22 = v24;
      v5[6] = v24;
      if ( v22 )
        sub_1623210((__int64)&v24, v22, (__int64)(v5 + 6));
    }
    return (__int64)v5;
  }
  if ( v7 == a3 )
    return (__int64)v5;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v26 = 257;
    v13 = 38;
    v14 = a3;
    goto LABEL_12;
  }
  v11 = (__int64 **)a3;
  v12 = 38;
  return sub_15A46C0(v12, (__int64 ***)a2, v11, 0);
}
