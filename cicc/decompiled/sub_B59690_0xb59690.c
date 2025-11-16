// Function: sub_B59690
// Address: 0xb59690
//
__int64 __fastcall sub_B59690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdx

  v4 = sub_B98A20(a2, a2, a3, a4);
  v5 = sub_BD5C60(a1, a2);
  result = sub_B9F6F0(v5, v4);
  v7 = a1 + 32 * (4LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v7 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    **(_QWORD **)(v7 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
  }
  *(_QWORD *)v7 = result;
  if ( result )
  {
    v9 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v7 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v7 + 8;
    *(_QWORD *)(v7 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v7;
  }
  return result;
}
