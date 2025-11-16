// Function: sub_222C7F0
// Address: 0x222c7f0
//
__int64 __fastcall sub_222C7F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  char v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 result; // rax

  if ( !sub_2207CD0((_QWORD *)(a1 + 104)) )
    return 0;
  v4 = sub_222BE90(a1, a2, v2, v3);
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 192) = 0;
  sub_222BC50(a1);
  *(_WORD *)(a1 + 169) = 0;
  v5 = *(_QWORD *)(a1 + 152);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 16) = v5;
  *(_QWORD *)(a1 + 24) = v5;
  v6 = *(_QWORD *)(a1 + 124);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 132) = v6;
  *(_QWORD *)(a1 + 140) = v6;
  if ( !sub_2207D40(a1 + 104) )
    return 0;
  result = a1;
  if ( v4 != 1 )
    return 0;
  return result;
}
