// Function: sub_CF0AE0
// Address: 0xcf0ae0
//
__int64 __fastcall sub_CF0AE0(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 *v6; // r13
  __int64 *v7; // rax
  __int64 v8; // r12
  _QWORD *v10; // rax
  _BYTE *v11; // rbx
  _QWORD *v12; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v13; // [rsp+10h] [rbp-40h] BYREF
  _BYTE *v14; // [rsp+18h] [rbp-38h]
  __int64 v15; // [rsp+20h] [rbp-30h]

  v13 = 0;
  v14 = 0;
  v15 = 0;
  v2 = (_QWORD *)sub_B2BE50(a1);
  v3 = sub_BCB2D0(v2);
  v4 = sub_ACD640(v3, a2, 0);
  if ( *(_BYTE *)v4 == 24 )
  {
    v5 = v14;
    v12 = *(_QWORD **)(v4 + 24);
  }
  else
  {
    v10 = sub_B98A20(v4, a2);
    v5 = v14;
    v12 = v10;
  }
  sub_914280((__int64)&v13, v5, &v12);
  v11 = v14;
  v6 = v13;
  v7 = (__int64 *)sub_B2BE50(a1);
  v8 = sub_B9C770(v7, v6, (__int64 *)((v11 - (_BYTE *)v6) >> 3), 0, 1);
  if ( v13 )
    j_j___libc_free_0(v13, v15 - (_QWORD)v13);
  return v8;
}
