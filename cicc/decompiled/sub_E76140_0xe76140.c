// Function: sub_E76140
// Address: 0xe76140
//
__int64 __fastcall sub_E76140(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 result; // rax

  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 1;
  *(_QWORD *)(a1 + 96) = a1;
  v3 = a1 + 112;
  *(_QWORD *)(v3 - 8) = 0;
  sub_C0BFB0(v3, 7, 0);
  v4 = *(_QWORD *)(a2 + 152);
  *(_BYTE *)(a1 + 160) = 0;
  result = *(unsigned __int8 *)(v4 + 348);
  *(_BYTE *)(a1 + 160) = result;
  if ( (_BYTE)result )
  {
    result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 168) + 104LL) + 16LL);
    *(_QWORD *)(a1 + 104) = result;
  }
  return result;
}
