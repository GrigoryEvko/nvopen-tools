// Function: sub_3175A30
// Address: 0x3175a30
//
__int64 __fastcall sub_3175A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rax
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  _BYTE *v11; // r13
  __int64 v12; // rax
  _BYTE *v14; // rsi

  v6 = *(_QWORD **)(a1 + 240);
  v7 = *(_BYTE **)(a2 - 96);
  if ( (_BYTE *)*v6 != v7 )
  {
    v11 = (_BYTE *)sub_31751A0(a1, v7);
    if ( v11 )
    {
      v12 = **(_QWORD **)(a1 + 240);
      if ( v12 == *(_QWORD *)(a2 - 64) )
      {
        if ( sub_AD7A80(v11, (__int64)v7, v8, v9, v10) )
          return *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL);
        v12 = **(_QWORD **)(a1 + 240);
      }
      if ( *(_QWORD *)(a2 - 32) == v12 && sub_AD7890((__int64)v11, (__int64)v7, v8, v9, v10) )
        return *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL);
    }
    return 0;
  }
  if ( sub_AD7890(v6[1], (__int64)v7, a3, a4, a5) )
    v14 = *(_BYTE **)(a2 - 32);
  else
    v14 = *(_BYTE **)(a2 - 64);
  return sub_31751A0(a1, v14);
}
