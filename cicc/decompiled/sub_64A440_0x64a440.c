// Function: sub_64A440
// Address: 0x64a440
//
__int64 __fastcall sub_64A440(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  char i; // al
  char v5; // al
  _QWORD *v6; // rax
  char v7; // bl
  _QWORD *v8; // r13
  __int64 v10; // rdi
  __int64 v11; // rdx

  v3 = *(_QWORD *)(a1 + 152);
  for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  if ( !i )
    return 1;
  v5 = *(_BYTE *)(a1 + 174);
  if ( v5 == 2 )
  {
    if ( unk_4D04880 )
      goto LABEL_19;
    return 1;
  }
  if ( v5 == 1 )
    goto LABEL_6;
LABEL_19:
  if ( (unsigned int)sub_8D4290(*(_QWORD *)(v3 + 160)) )
  {
LABEL_6:
    v6 = *(_QWORD **)(v3 + 168);
    v7 = 0;
    v8 = (_QWORD *)*v6;
    if ( (*(_BYTE *)(a1 + 193) & 5) != 0
      && (*(_BYTE *)(a1 + 194) & 0x40) == 0
      && (*(_BYTE *)(a1 + 195) & 0xB) != 1
      && (dword_4F077C4 != 2 || unk_4F07778 <= 202301) )
    {
      v7 = 1;
      if ( HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 )
          v7 = ((*(_BYTE *)(a1 + 206) >> 1) ^ 1) & 1;
      }
    }
    while ( v8 )
    {
      if ( !(unsigned int)sub_8D4290(v8[1]) )
      {
        if ( !v7 )
          return 0;
        sub_685360(2392, a2);
        return 0;
      }
      v8 = (_QWORD *)*v8;
    }
    return 1;
  }
  if ( (*(_BYTE *)(a1 + 193) & 5) == 0
    || (*(_BYTE *)(a1 + 195) & 0xB) == 1
    || dword_4F077C4 == 2 && unk_4F07778 > 202301
    || HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 && qword_4F077A8 && (*(_BYTE *)(a1 + 206) & 2) != 0 )
  {
    return 0;
  }
  v10 = 8;
  if ( *(char *)(a1 + 202) < 0 )
  {
    v11 = qword_4F04C68[0] + 776LL * dword_4F04C64 - 776;
    if ( *(_BYTE *)(v11 + 4) == 7 && (*(_BYTE *)(*(_QWORD *)(v11 + 208) + 177LL) & 0x10) != 0 )
      v10 = byte_4F07472[0];
  }
  sub_685260(v10, 2391, a2, *(_QWORD *)(v3 + 160));
  return 0;
}
