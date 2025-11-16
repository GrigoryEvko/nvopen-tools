// Function: sub_18DE3D0
// Address: 0x18de3d0
//
__int64 __fastcall sub_18DE3D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r14
  __int64 v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx

  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  if ( *(_BYTE *)(a3 + 16) == 79 && *(_QWORD *)(a2 - 72) == *(_QWORD *)(a3 - 72) )
  {
    if ( !(unsigned __int8)sub_18DDD00(a1, *(_QWORD *)(a2 - 48), *(_QWORD *)(a3 - 48), v5) )
    {
      v9 = *(_QWORD *)(a3 - 24);
      v7 = *(_QWORD *)(a2 - 24);
      v8 = v5;
      return sub_18DDD00(a1, v7, v9, v8);
    }
    return 1;
  }
  if ( (unsigned __int8)sub_18DDD00(a1, *(_QWORD *)(a2 - 48), a3, v5) )
    return 1;
  v7 = *(_QWORD *)(a2 - 24);
  v8 = v5;
  v9 = a3;
  return sub_18DDD00(a1, v7, v9, v8);
}
