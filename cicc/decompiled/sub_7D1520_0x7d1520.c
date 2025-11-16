// Function: sub_7D1520
// Address: 0x7d1520
//
__int64 __fastcall sub_7D1520(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  char v5; // dl
  __int64 result; // rax

  v5 = *(_BYTE *)(a1 + 4);
  result = a2;
  if ( (!v5 || (unsigned __int8)(v5 - 3) <= 2u)
    && *(_QWORD *)(a1 + 544)
    && (!a4[8] || unk_4D047B4 && (!dword_4F077BC || qword_4F077A8 <= 0x9C3Fu || !a2)) )
  {
    return sub_7D1000(a1, a2, a3, a4);
  }
  return result;
}
