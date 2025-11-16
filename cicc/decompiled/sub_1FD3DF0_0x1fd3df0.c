// Function: sub_1FD3DF0
// Address: 0x1fd3df0
//
char *__fastcall sub_1FD3DF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r10
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  int v12; // eax
  _BYTE v14[32]; // [rsp+10h] [rbp-20h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 784LL);
  v6 = *(_QWORD *)(v5 + 40);
  v7 = v6 + 40;
  v8 = *(_QWORD *)(v6 + 48);
  if ( v8 == v7 )
    goto LABEL_5;
  v9 = 0;
  do
  {
    v8 = *(_QWORD *)(v8 + 8);
    ++v9;
  }
  while ( v8 != v7 );
  if ( v9 == 1 )
    goto LABEL_5;
  if ( !sub_1DD69A0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 784LL), a2) )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 784LL);
LABEL_5:
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 104)
                                                                                               + 288LL))(
      *(_QWORD *)(a1 + 104),
      v5,
      a2,
      0,
      v14,
      0,
      a3,
      0);
  }
  v10 = *(_QWORD *)(a1 + 40);
  v11 = *(_QWORD *)(v10 + 32);
  if ( !v11 )
    return sub_1DD8D40(*(_QWORD *)(v10 + 784), a2);
  v12 = sub_13774B0(v11, *(_QWORD *)(*(_QWORD *)(v10 + 784) + 40LL), *(_QWORD *)(a2 + 40));
  return sub_1DD8FE0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 784LL), a2, v12);
}
