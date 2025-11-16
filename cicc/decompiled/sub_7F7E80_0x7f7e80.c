// Function: sub_7F7E80
// Address: 0x7f7e80
//
_BOOL8 __fastcall sub_7F7E80(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r12

  v1 = unk_4F072B8 + 16LL * *(int *)(a1 + 160);
  if ( !*(_QWORD *)(unk_4F073B0 + 8LL * *(int *)(v1 + 8)) )
    return 0;
  v2 = *(_QWORD *)v1;
  if ( !*(_QWORD *)v1 )
    return 0;
  v3 = *(_QWORD *)(v2 + 80);
  v4 = 0;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    v4 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  return *(_BYTE *)(v3 + 40) == 11
      && (unsigned int)sub_7F7BA0(*(_QWORD *)(v3 + 72))
      && ((*(_BYTE *)(v2 + 29) & 1) != 0 || !v4 || (*(_WORD *)(v4 + 176) & 0x110) == 0);
}
