// Function: sub_1284400
// Address: 0x1284400
//
__int64 __fastcall sub_1284400(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // r12
  unsigned __int64 *v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v19[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v20; // [rsp+20h] [rbp-30h]

  v8 = sub_15A0680(*(_QWORD *)a2, a3, 0);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v8 + 16) <= 0x10u )
    return sub_15A2D80(a2, v8, a5);
  v20 = 257;
  if ( a5 )
  {
    v12 = (_QWORD *)sub_15FB440(24, a2, v8, v19, 0);
    sub_15F2350(v12, 1);
    v11 = a1[1];
    if ( !v11 )
      goto LABEL_7;
    goto LABEL_6;
  }
  v10 = sub_15FB440(24, a2, v8, v19, 0);
  v11 = a1[1];
  v12 = (_QWORD *)v10;
  if ( v11 )
  {
LABEL_6:
    v13 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v12);
    v14 = v12[3];
    v15 = *v13;
    v12[4] = v13;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v15 | v14 & 7;
    *(_QWORD *)(v15 + 8) = v12 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v12 + 3);
  }
LABEL_7:
  sub_164B780(v12, a4);
  v16 = *a1;
  if ( *a1 )
  {
    v18 = *a1;
    sub_1623A60(&v18, v16, 2);
    if ( v12[6] )
      sub_161E7C0(v12 + 6);
    v17 = v18;
    v12[6] = v18;
    if ( v17 )
      sub_1623210(&v18, v17, v12 + 6);
  }
  return (__int64)v12;
}
