// Function: sub_38723F0
// Address: 0x38723f0
//
_QWORD *__fastcall sub_38723F0(__int64 *a1, int a2, __int64 a3, __int64 **a4, __int64 *a5)
{
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 *v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  unsigned __int8 *v18; // [rsp+8h] [rbp-48h] BYREF
  char v19[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v20; // [rsp+20h] [rbp-30h]

  if ( a4 == *(__int64 ***)a3 )
    return (_QWORD *)a3;
  if ( *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v20 = 257;
    v10 = sub_15FDBD0(a2, a3, (__int64)a4, (__int64)v19, 0);
    v11 = a1[1];
    v6 = (_QWORD *)v10;
    if ( v11 )
    {
      v12 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v11 + 40, v10);
      v13 = v6[3];
      v14 = *v12;
      v6[4] = v12;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      v6[3] = v14 | v13 & 7;
      *(_QWORD *)(v14 + 8) = v6 + 3;
      *v12 = *v12 & 7 | (unsigned __int64)(v6 + 3);
    }
    sub_164B780((__int64)v6, a5);
    v15 = *a1;
    if ( *a1 )
    {
      v18 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v18, v15, 2);
      v16 = v6[6];
      if ( v16 )
        sub_161E7C0((__int64)(v6 + 6), v16);
      v17 = v18;
      v6[6] = v18;
      if ( v17 )
        sub_1623210((__int64)&v18, v17, (__int64)(v6 + 6));
    }
  }
  else
  {
    v6 = (_QWORD *)sub_15A46C0(a2, (__int64 ***)a3, a4, 0);
    v7 = sub_14DBA30((__int64)v6, a1[8], 0);
    if ( v7 )
      return (_QWORD *)v7;
  }
  return v6;
}
