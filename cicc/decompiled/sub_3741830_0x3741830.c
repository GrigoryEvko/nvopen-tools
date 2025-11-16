// Function: sub_3741830
// Address: 0x3741830
//
__int64 __fastcall sub_3741830(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v21; // [rsp-8h] [rbp-38h]
  _BYTE v22[32]; // [rsp+10h] [rbp-20h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(v7 + 56);
  v9 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 )
  {
    if ( v9 && v8 == v9 )
      goto LABEL_4;
  }
  else if ( !v9 )
  {
LABEL_4:
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 120)
                                                                                               + 368LL))(
      *(_QWORD *)(a1 + 120),
      v6,
      a2,
      0,
      v22,
      0,
      a3,
      0);
    v12 = v21;
    goto LABEL_5;
  }
  if ( !*(_BYTE *)(sub_AA4B30(*(_QWORD *)(v6 + 16)) + 872) && sub_AA6A60(v7) <= 1
    || !sub_2E322F0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL), a2) )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL);
    goto LABEL_4;
  }
LABEL_5:
  v13 = *(_QWORD *)(a1 + 40);
  v14 = *(_QWORD *)(v13 + 32);
  v15 = *(_QWORD *)(v13 + 744);
  if ( !v14 )
    return sub_2E321B0(*(_QWORD *)(v13 + 744), a2, v12, v10, v15, v11);
  v16 = sub_FF0430(v14, *(_QWORD *)(v15 + 16), *(_QWORD *)(a2 + 16));
  return sub_2E33F80(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL), a2, v16, v17, v18, v19);
}
