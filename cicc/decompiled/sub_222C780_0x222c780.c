// Function: sub_222C780
// Address: 0x222c780
//
__int64 __fastcall sub_222C780(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 result; // rax

  v2 = *a1;
  *(_DWORD *)(v2 + 120) = 0;
  *(_BYTE *)(v2 + 192) = 0;
  sub_222BC50(v2);
  v3 = *a1;
  *(_WORD *)(v3 + 169) = 0;
  v4 = *(_QWORD *)(v3 + 152);
  *(_QWORD *)(v3 + 40) = 0;
  *(_QWORD *)(v3 + 8) = v4;
  *(_QWORD *)(v3 + 16) = v4;
  *(_QWORD *)(v3 + 24) = v4;
  v5 = *(_QWORD *)(v3 + 124);
  *(_QWORD *)(v3 + 32) = 0;
  *(_QWORD *)(v3 + 132) = v5;
  *(_QWORD *)(v3 + 48) = 0;
  result = *a1;
  *(_QWORD *)(*a1 + 140) = *(_QWORD *)(*a1 + 132);
  return result;
}
