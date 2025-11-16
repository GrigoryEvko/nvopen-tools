// Function: sub_3797A70
// Address: 0x3797a70
//
__int64 __fastcall sub_3797A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // r13d
  bool v10; // al
  unsigned __int16 *v12; // rdx
  _QWORD *v13; // r12
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r8
  unsigned __int16 v17; // ax
  _QWORD *v18; // r12
  __int64 v19; // rdx
  __int16 v20; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+10h] [rbp-30h] BYREF
  int v23; // [rsp+18h] [rbp-28h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = **(_QWORD **)(*(_QWORD *)(v6 + 80) + 40LL);
  if ( *(_DWORD *)(v7 + 24) == 51 )
  {
    v12 = *(unsigned __int16 **)(a2 + 48);
    v13 = *(_QWORD **)(a1 + 8);
    v14 = *v12;
    v15 = *((_QWORD *)v12 + 1);
    v20 = v14;
    v21 = v15;
    if ( (_WORD)v14 )
    {
      v16 = 0;
      v17 = word_4456580[v14 - 1];
    }
    else
    {
      v17 = sub_3009970((__int64)&v20, a2, v15, a4, a5);
      v16 = v19;
    }
    v22 = 0;
    v23 = 0;
    v18 = sub_33F17F0(v13, 51, (__int64)&v22, v17, v16);
    if ( v22 )
      sub_B91220((__int64)&v22, v22);
    return (__int64)v18;
  }
  else
  {
    v8 = *(_QWORD *)(v7 + 96);
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v8 + 24) == 0;
    else
      v10 = v9 == (unsigned int)sub_C444A0(v8 + 24);
    return sub_37946F0(a1, *(_QWORD *)(v6 + 40LL * !v10), *(_QWORD *)(v6 + 40LL * !v10 + 8));
  }
}
