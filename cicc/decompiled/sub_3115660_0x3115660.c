// Function: sub_3115660
// Address: 0x3115660
//
__int64 __fastcall sub_3115660(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r13
  bool v6; // r8
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  char v11; // [rsp+Ch] [rbp-34h]

  result = sub_31155C0(a1, (unsigned int *)a2);
  if ( v4 )
  {
    v5 = (_QWORD *)v4;
    v6 = 1;
    if ( !result && v4 != a1 + 8 )
      v6 = *(_DWORD *)a2 < *(_DWORD *)(v4 + 32);
    v11 = v6;
    v7 = sub_22077B0(0x50u);
    *(_DWORD *)(v7 + 32) = *(_DWORD *)a2;
    *(_QWORD *)(v7 + 40) = *(_QWORD *)(a2 + 8);
    *(_DWORD *)(v7 + 48) = *(_DWORD *)(a2 + 16);
    v8 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 24) = 0;
    *(_QWORD *)(v7 + 56) = v8;
    v9 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 32) = 0;
    *(_QWORD *)(v7 + 64) = v9;
    v10 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 40) = 0;
    *(_QWORD *)(v7 + 72) = v10;
    sub_220F040(v11, v7, v5, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
    return v7;
  }
  return result;
}
