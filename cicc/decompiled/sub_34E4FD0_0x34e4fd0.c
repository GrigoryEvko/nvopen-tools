// Function: sub_34E4FD0
// Address: 0x34e4fd0
//
__int64 __fastcall sub_34E4FD0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  const void *v3; // r9
  size_t v4; // r12
  size_t v5; // rdx
  unsigned int v6; // r15d
  _QWORD *v7; // rax
  __int64 (*v8)(); // rax
  __int64 v9; // rsi
  _QWORD *v10; // r15
  __int64 *v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdi
  void *src; // [rsp+0h] [rbp-90h]
  _QWORD *v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int8 *v21; // [rsp+18h] [rbp-78h] BYREF
  __int64 v22; // [rsp+20h] [rbp-70h] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h]
  __int64 v24; // [rsp+30h] [rbp-60h]
  unsigned __int64 v25[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+50h] [rbp-40h] BYREF

  v21 = (unsigned __int8 *)sub_B2D7E0(*(_QWORD *)a2, "fentry-call", 0xBu);
  v2 = (_BYTE *)sub_A72240((__int64 *)&v21);
  v25[0] = (unsigned __int64)v26;
  v3 = v2;
  v4 = v5;
  LOBYTE(v5) = v2 == 0 && &v2[v5] != 0;
  if ( (_BYTE)v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v22 = v4;
  v6 = v5;
  if ( v4 > 0xF )
  {
    src = v2;
    v16 = sub_22409D0((__int64)v25, (unsigned __int64 *)&v22, 0);
    v3 = src;
    v25[0] = v16;
    v17 = (_QWORD *)v16;
    v26[0] = v22;
  }
  else
  {
    if ( v4 == 1 )
    {
      LOBYTE(v26[0]) = *v2;
      v7 = v26;
      goto LABEL_5;
    }
    if ( !v4 )
    {
      v7 = v26;
      goto LABEL_5;
    }
    v17 = v26;
  }
  memcpy(v17, v3, v4);
  v4 = v22;
  v7 = (_QWORD *)v25[0];
LABEL_5:
  v25[1] = v4;
  *((_BYTE *)v7 + v4) = 0;
  if ( !sub_2241AC0((__int64)v25, "true") )
  {
    v8 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 128LL);
    if ( v8 == sub_2DAC790 )
      BUG();
    v19 = *(_QWORD **)(a2 + 328);
    v20 = 0;
    v9 = *(_QWORD *)(v8() + 8);
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v10 = (_QWORD *)v19[4];
    v11 = (__int64 *)v19[7];
    v21 = 0;
    v12 = (__int64)sub_2E7B380(v10, v9 - 1080, &v21, 0);
    if ( v21 )
      sub_B91220((__int64)&v21, (__int64)v21);
    sub_2E31040(v19 + 5, v12);
    v13 = *v11;
    v14 = *(_QWORD *)v12;
    *(_QWORD *)(v12 + 8) = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v12 = v13 | v14 & 7;
    *(_QWORD *)(v13 + 8) = v12;
    *v11 = v12 | *v11 & 7;
    if ( v23 )
      sub_2E882B0(v12, (__int64)v10, v23);
    if ( v24 )
      sub_2E88680(v12, (__int64)v10, v24);
    if ( v22 )
      sub_B91220((__int64)&v22, v22);
    if ( v20 )
      sub_B91220((__int64)&v20, v20);
    v6 = 1;
  }
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0]);
  return v6;
}
