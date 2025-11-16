// Function: sub_1484A80
// Address: 0x1484a80
//
__int64 __fastcall sub_1484A80(_QWORD *a1, __int64 a2, __int64 *a3, __int64 *a4, __m128i a5, __m128i a6)
{
  bool v6; // zf
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-68h] BYREF
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v19; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v20; // [rsp+28h] [rbp-48h]
  _QWORD *v21; // [rsp+30h] [rbp-40h]
  __int64 *v22; // [rsp+38h] [rbp-38h]
  __int64 *v23; // [rsp+40h] [rbp-30h]

  v6 = *(_WORD *)(a2 + 24) == 4;
  v17 = a2;
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(a2 + 40) != 2 )
    return 0;
  v9 = *(_QWORD **)(a2 + 32);
  v10 = v9[1];
  v18 = v10;
  v11 = *v9;
  if ( *(_WORD *)(*v9 + 24LL) != 5 )
    return 0;
  v21 = a1;
  v19 = &v17;
  v20 = &v18;
  v12 = *(_QWORD *)(v11 + 40);
  v22 = a3;
  v23 = a4;
  if ( v12 == 3 )
  {
    v14 = *(_QWORD **)(v11 + 32);
    if ( !*(_WORD *)(*v14 + 24LL) )
    {
      v13 = v14[1];
      if ( a2 != sub_1484870(a1, v10, v13, a5, a6) )
        return sub_1484A10(&v19, *(_QWORD *)(*(_QWORD *)(v11 + 32) + 16LL), a5, a6);
      goto LABEL_13;
    }
    return 0;
  }
  if ( v12 != 2 )
    return 0;
  v13 = *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8LL);
  if ( a2 == sub_1484870(a1, v10, v13, a5, a6) )
  {
LABEL_13:
    *v22 = *v20;
    *v23 = v13;
    return 1;
  }
  if ( (unsigned __int8)sub_1484A10(&v19, **(_QWORD **)(v11 + 32), a5, a6) )
    return 1;
  v15 = sub_1480620((__int64)a1, *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8LL), 0);
  if ( (unsigned __int8)sub_1484A10(&v19, v15, a5, a6) )
    return 1;
  v16 = sub_1480620((__int64)a1, **(_QWORD **)(v11 + 32), 0);
  return sub_1484A10(&v19, v16, a5, a6);
}
