// Function: sub_13AC120
// Address: 0x13ac120
//
__int64 __fastcall sub_13AC120(__int64 a1, __int64 a2, __int64 a3, int *a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  __int16 v10; // ax
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD **v14; // r15
  int v15; // eax
  char v16; // al
  __int64 v18; // r15
  int v19; // eax
  __int64 v20; // r15
  int v21; // eax
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+28h] [rbp-48h]
  _QWORD **v27; // [rsp+28h] [rbp-48h]
  _QWORD **v28; // [rsp+28h] [rbp-48h]

  v10 = *(_WORD *)(a3 + 24);
  if ( *(_WORD *)(a2 + 24) == 7 )
  {
    v11 = *(__int64 **)(a2 + 32);
    v12 = *(_QWORD *)(a1 + 8);
    v24 = *v11;
    if ( v10 == 7 )
    {
      v22 = **(_QWORD **)(a3 + 32);
      v26 = sub_13A5BC0((_QWORD *)a2, v12);
      v13 = sub_13A5BC0((_QWORD *)a3, *(_QWORD *)(a1 + 8));
      v14 = *(_QWORD ***)(a2 + 48);
      v23 = v13;
      v15 = sub_13A6B70(a1, v14);
      *a4 = v15;
      if ( v26 == v23 )
      {
        v16 = sub_13A83F0(a1, v26, v24, v22, (__int64)v14, v15, a5, a6);
      }
      else if ( v26 == sub_1480620(*(_QWORD *)(a1 + 8), v23, 0) )
      {
        v16 = sub_13A88D0(a1, v26, v24, v22, (__int64)v14, *a4, a5, a6, a7);
      }
      else
      {
        v16 = sub_13A8D80(a1, v26, v23, v24, v22, (__int64)v14, *a4, a5, a6);
      }
      if ( !v16 && !sub_13AB330(a1, a2, a3, a5) )
        return sub_13AB010(a1, v26, v23, v24, v22, (__int64)v14, (__int64)v14);
      return 1;
    }
    v18 = sub_13A5BC0((_QWORD *)a2, v12);
    v27 = *(_QWORD ***)(a2 + 48);
    v19 = sub_13A6B70(a1, v27);
    *a4 = v19;
    if ( (unsigned __int8)sub_13AA1C0(a1, v18, v24, a3, (__int64)v27, v19, a5, a6) )
      return 1;
  }
  else
  {
    v25 = **(_QWORD **)(a3 + 32);
    v20 = sub_13A5BC0((_QWORD *)a3, *(_QWORD *)(a1 + 8));
    v28 = *(_QWORD ***)(a3 + 48);
    v21 = sub_13A6B90(a1, v28);
    *a4 = v21;
    if ( (unsigned __int8)sub_13A9F60(a1, v20, a2, v25, (__int64)v28, v21, a5, a6) )
      return 1;
  }
  return sub_13AB330(a1, a2, a3, a5);
}
