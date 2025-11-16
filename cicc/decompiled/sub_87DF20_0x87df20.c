// Function: sub_87DF20
// Address: 0x87df20
//
__int64 __fastcall sub_87DF20(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  _QWORD *v3; // r14
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  char v7; // al

  if ( (*(_BYTE *)(a1 + 97) & 1) == 0 )
  {
    v1 = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)(a1 + 96) & 2) != 0 )
      return sub_87DE40(a1, *(_QWORD *)(a1 + 56));
    v2 = *(_QWORD *)(a1 + 112);
    v3 = *(_QWORD **)(v2 + 16);
    v4 = *(_QWORD **)(v2 + 8);
    if ( (_QWORD *)*v3 == v4 )
      return 1;
    while ( 1 )
    {
      v6 = v4[2];
      v7 = *(_BYTE *)(v6 + 96);
      if ( (v7 & 2) == 0 )
        break;
      if ( (v7 & 1) != 0 )
      {
        v5 = *(_QWORD *)(v6 + 112);
        if ( !*(_QWORD *)v5 )
          goto LABEL_15;
      }
      if ( (unsigned int)sub_87DE40(v4[2], v1) )
        goto LABEL_12;
LABEL_8:
      if ( *(_BYTE *)(*(_QWORD *)(v6 + 112) + 25LL) != 1
        || !dword_4F077BC
        || qword_4F077A8 > 0x9DCFu
        || !(unsigned int)sub_87E070(v6, a1) )
      {
        return 0;
      }
LABEL_12:
      v1 = *(_QWORD *)(v6 + 40);
      v4 = (_QWORD *)*v4;
      if ( (_QWORD *)*v3 == v4 )
        return 1;
    }
    v5 = *(_QWORD *)(v6 + 112);
LABEL_15:
    if ( !*(_BYTE *)(v5 + 25) || (unsigned int)sub_87D890(v1) )
      goto LABEL_12;
    if ( *(_BYTE *)(*(_QWORD *)(v6 + 112) + 25LL) != 1 )
      return 0;
    if ( (unsigned int)sub_87D970(v1) )
      goto LABEL_12;
    goto LABEL_8;
  }
  return 1;
}
