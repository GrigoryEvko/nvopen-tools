// Function: sub_5CAE10
// Address: 0x5cae10
//
__int64 __fastcall sub_5CAE10(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 i; // rbx
  char v4; // al

  v2 = *(_QWORD *)(a2 + 120);
  for ( i = *(_QWORD *)(a1 + 48); *(_BYTE *)(v2 + 140) == 12; v2 = *(_QWORD *)(v2 + 160) )
    ;
  if ( (unsigned int)sub_8D3410(v2) )
    v2 = sub_8D40F0(v2);
  if ( (unsigned int)sub_8D3A70(v2)
    && (*(_BYTE *)(i + 122) & 1) != 0
    && ((v4 = *(_BYTE *)(unk_4F04C68 + 776LL * unk_4F04C64 + 4), (unsigned __int8)(v4 - 3) <= 1u)
     || !v4
     || (*(_BYTE *)(a2 + 89) & 4) != 0) )
  {
    sub_5CAD90(a1, 1u, (_WORD *)(a2 + 158));
    return a2;
  }
  else
  {
    sub_6851C0(1219, a1 + 56);
    return a2;
  }
}
