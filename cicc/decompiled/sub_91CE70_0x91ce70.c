// Function: sub_91CE70
// Address: 0x91ce70
//
_QWORD *__fastcall sub_91CE70(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rax
  unsigned __int64 v5[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 128LL);
  if ( !v2 )
    sub_91B8A0("label for goto statement not found!", (_DWORD *)a2, 1);
  if ( *(_BYTE *)(v2 + 40) != 7 || (*(_BYTE *)(*(_QWORD *)(v2 + 72) + 120LL) & 0xA) == 0 )
    *(_BYTE *)a1 = 1;
  v3 = *(_QWORD *)(a1 + 16);
  v5[0] = a2;
  v5[1] = v3;
  return sub_91CD50((_QWORD *)(a1 + 72), v5);
}
