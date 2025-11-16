// Function: sub_182E520
// Address: 0x182e520
//
__int64 __fastcall sub_182E520(__int64 **a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // r12
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  __int64 v16; // [rsp+10h] [rbp-88h] BYREF
  char v17[16]; // [rsp+18h] [rbp-80h] BYREF
  __int16 v18; // [rsp+28h] [rbp-70h]
  __int64 v19; // [rsp+38h] [rbp-60h] BYREF
  __int16 v20; // [rsp+48h] [rbp-50h]
  char v21[16]; // [rsp+58h] [rbp-40h] BYREF
  __int16 v22; // [rsp+68h] [rbp-30h]

  v4 = *a2;
  v20 = 257;
  v18 = 257;
  v5 = sub_1285290(a3, *(_QWORD *)(v4 + 24), (int)a2, 0, 0, (__int64)v17, 0);
  v6 = (_QWORD *)v5;
  if ( a1 != *(__int64 ***)v5 )
  {
    if ( *(_BYTE *)(v5 + 16) > 0x10u )
    {
      v22 = 257;
      v8 = sub_15FDBD0(37, v5, (__int64)a1, (__int64)v21, 0);
      v9 = a3[1];
      v6 = (_QWORD *)v8;
      if ( v9 )
      {
        v10 = (unsigned __int64 *)a3[2];
        sub_157E9D0(v9 + 40, v8);
        v11 = v6[3];
        v12 = *v10;
        v6[4] = v10;
        v12 &= 0xFFFFFFFFFFFFFFF8LL;
        v6[3] = v12 | v11 & 7;
        *(_QWORD *)(v12 + 8) = v6 + 3;
        *v10 = *v10 & 7 | (unsigned __int64)(v6 + 3);
      }
      sub_164B780((__int64)v6, &v19);
      v13 = *a3;
      if ( *a3 )
      {
        v16 = *a3;
        sub_1623A60((__int64)&v16, v13, 2);
        v14 = v6[6];
        if ( v14 )
          sub_161E7C0((__int64)(v6 + 6), v14);
        v15 = (unsigned __int8 *)v16;
        v6[6] = v16;
        if ( v15 )
          sub_1623210((__int64)&v16, v15, (__int64)(v6 + 6));
      }
    }
    else
    {
      return sub_15A46C0(37, (__int64 ***)v5, a1, 0);
    }
  }
  return (__int64)v6;
}
