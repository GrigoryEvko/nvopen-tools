// Function: sub_B59C10
// Address: 0xb59c10
//
__int64 __fastcall sub_B59C10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rdx

  v2 = sub_BD5C60(a1, a2);
  v3 = sub_BCB2D0(v2);
  result = sub_ACD640(v3, (unsigned int)a2, 0);
  v5 = a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v5 )
  {
    v6 = *(_QWORD *)(v5 + 8);
    **(_QWORD **)(v5 + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 + 16);
  }
  *(_QWORD *)v5 = result;
  if ( result )
  {
    v7 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v5 + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = v5 + 8;
    *(_QWORD *)(v5 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v5;
  }
  return result;
}
