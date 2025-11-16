// Function: sub_13AC390
// Address: 0x13ac390
//
__int64 __fastcall sub_13AC390(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v5; // ax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r13
  char v9; // al
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+20h] [rbp-40h]

  v5 = *(_WORD *)(a3 + 24);
  if ( *(_WORD *)(a2 + 24) == 7 )
  {
    v6 = *(__int64 **)(a2 + 32);
    if ( v5 == 7 )
    {
      v22 = *v6;
      v21 = sub_13A5BC0((_QWORD *)a2, *(_QWORD *)(a1 + 8));
      v20 = *(_QWORD *)(a2 + 48);
      v7 = **(_QWORD **)(a3 + 32);
      v8 = sub_13A5BC0((_QWORD *)a3, *(_QWORD *)(a1 + 8));
      v19 = *(_QWORD *)(a3 + 48);
      v9 = sub_13AA420(a1, v21, v8, v22, v7, v20, v19, a4);
    }
    else
    {
      v11 = *v6;
      v22 = **(_QWORD **)(*v6 + 32);
      v12 = sub_13A5BC0((_QWORD *)*v6, *(_QWORD *)(a1 + 8));
      v13 = *(_QWORD *)(a1 + 8);
      v21 = v12;
      v14 = *(_QWORD *)(v11 + 48);
      v7 = a3;
      v20 = v14;
      v15 = sub_13A5BC0((_QWORD *)a2, v13);
      v8 = sub_1480620(v13, v15, 0);
      v19 = *(_QWORD *)(a2 + 48);
      v9 = sub_13AA420(a1, v21, v8, v22, a3, v20, v19, a4);
    }
  }
  else
  {
    v23 = **(_QWORD **)(a3 + 32);
    v7 = **(_QWORD **)(v23 + 32);
    v8 = sub_13A5BC0((_QWORD *)v23, *(_QWORD *)(a1 + 8));
    v16 = *(_QWORD *)(v23 + 48);
    v24 = *(_QWORD *)(a1 + 8);
    v19 = v16;
    v17 = sub_13A5BC0((_QWORD *)a3, v24);
    v18 = sub_1480620(v24, v17, 0);
    v22 = a2;
    v21 = v18;
    v20 = *(_QWORD *)(a3 + 48);
    v9 = sub_13AA420(a1, v18, v8, a2, v7, v20, v19, a4);
  }
  if ( v9 || sub_13AB330(a1, a2, a3, a4) )
    return 1;
  else
    return sub_13AB010(a1, v21, v8, v22, v7, v20, v19);
}
