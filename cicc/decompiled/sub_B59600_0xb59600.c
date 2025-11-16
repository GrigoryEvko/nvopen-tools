// Function: sub_B59600
// Address: 0xb59600
//
__int64 __fastcall sub_B59600(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rdx

  v2 = sub_BD5C60(a1, a2);
  result = sub_B9F6F0(v2, a2);
  v4 = a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    **(_QWORD **)(v4 + 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v4 + 16);
  }
  *(_QWORD *)v4 = result;
  if ( result )
  {
    v6 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v4 + 8) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = v4 + 8;
    *(_QWORD *)(v4 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v4;
  }
  return result;
}
