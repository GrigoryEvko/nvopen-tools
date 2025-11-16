// Function: sub_2D27AA0
// Address: 0x2d27aa0
//
unsigned __int8 *__fastcall sub_2D27AA0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // rax
  unsigned __int8 *v6; // r13
  _QWORD *v8; // rax
  unsigned int v9; // [rsp+Ch] [rbp-84h]
  __int64 v10; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v11; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-68h]
  _QWORD *v13; // [rsp+30h] [rbp-60h] BYREF
  __int64 v14; // [rsp+38h] [rbp-58h]
  _QWORD v15[10]; // [rsp+40h] [rbp-50h] BYREF

  v13 = (_QWORD *)sub_9208B0(a1, *(_QWORD *)(a2 + 8));
  v14 = v4;
  v12 = sub_CA1930(&v13);
  if ( v12 > 0x40 )
    sub_C43690((__int64)&v11, 0, 0);
  else
    v11 = 0;
  v5 = sub_BD45C0((unsigned __int8 *)a2, a1, (__int64)&v11, 0, 0, 0, 0, 0);
  v13 = v15;
  v6 = v5;
  v14 = 0x300000000LL;
  if ( v12 > 0x40 )
  {
    v9 = v12;
    if ( v9 == (unsigned int)sub_C444A0((__int64)&v11) )
      goto LABEL_5;
    v8 = (_QWORD *)*v11;
    goto LABEL_12;
  }
  v8 = v11;
  if ( v11 )
  {
LABEL_12:
    v15[0] = 35;
    v15[1] = v8;
    LODWORD(v14) = 2;
    a3 = (_QWORD *)sub_B0D8A0(a3, (__int64)&v13, 0, 0);
  }
LABEL_5:
  v10 = 6;
  sub_B0DED0(a3, &v10, 1);
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0((unsigned __int64)v11);
  return v6;
}
