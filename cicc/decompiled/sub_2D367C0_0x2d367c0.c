// Function: sub_2D367C0
// Address: 0x2d367c0
//
void __fastcall sub_2D367C0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int8 *v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // rax
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  _QWORD *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-88h]
  unsigned int v21[4]; // [rsp+20h] [rbp-70h] BYREF
  char v22; // [rsp+30h] [rbp-60h]
  __int64 v23[10]; // [rsp+40h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a3 + 24);
  v23[0] = v7;
  if ( v7 )
    sub_B96E90((__int64)v23, v7, 1);
  v8 = sub_B10CD0((__int64)v23);
  if ( v23[0] )
  {
    v20 = v8;
    sub_B91220((__int64)v23, v23[0]);
    v8 = v20;
  }
  v23[2] = a4;
  v9 = a3 + 80;
  v23[0] = a1;
  v23[1] = a3;
  v23[3] = v8;
  if ( a2 )
  {
    if ( a2 != 1 )
    {
      v10 = sub_B11F60(a3 + 80);
      sub_2D36680(v23, 0, v10);
      return;
    }
    goto LABEL_9;
  }
  v11 = sub_2D28490(a3);
  if ( (unsigned __int8)sub_B14070(v11) )
  {
LABEL_9:
    v12 = sub_B11F60(a3 + 80);
    sub_2D36680(v23, *(_QWORD **)(a3 + 40), v12);
    return;
  }
  v13 = sub_B13320(v11);
  v14 = (_QWORD *)sub_B11F60(v11 + 88);
  v15 = sub_B11F60(v9);
  sub_AF47B0((__int64)v21, *(unsigned __int64 **)(v15 + 16), *(unsigned __int64 **)(v15 + 24));
  if ( v22 )
    v14 = (_QWORD *)sub_B0E470((__int64)v14, v21[2], v21[0]);
  v16 = sub_2D27AA0(*(_QWORD *)(a1 + 128), (__int64)v13, v14);
  v18 = v17;
  v19 = sub_B98A20((__int64)v16, (__int64)v13);
  sub_2D36680(v23, v19, v18);
}
