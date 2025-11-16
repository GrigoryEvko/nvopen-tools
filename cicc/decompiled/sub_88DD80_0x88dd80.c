// Function: sub_88DD80
// Address: 0x88dd80
//
void __fastcall sub_88DD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // r13

  v6 = qword_4F04C68[0] + 776LL * *(int *)(a1 + 204);
  v7 = *(unsigned __int8 *)(v6 + 4);
  if ( (unsigned __int8)(v7 - 3) <= 1u )
  {
    sub_877E90(a2, 0, *(_QWORD *)(*(_QWORD *)(v6 + 184) + 32LL));
  }
  else if ( (_BYTE)v7 == 6 )
  {
    v8 = *(_QWORD *)(a2 + 88);
    sub_877E20(a2, 0, *(_QWORD *)(a1 + 240), (unsigned int)(v7 - 3), a5, a6);
    if ( *(_BYTE *)(a2 + 80) == 19 )
      *(_BYTE *)(v8 + 265) = (*(_BYTE *)(a1 + 164) << 6) | *(_BYTE *)(v8 + 265) & 0x3F;
  }
}
