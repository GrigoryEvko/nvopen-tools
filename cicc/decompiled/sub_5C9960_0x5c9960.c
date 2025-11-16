// Function: sub_5C9960
// Address: 0x5c9960
//
__int64 __fastcall sub_5C9960(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r13
  __int64 v6; // rdx
  char *v7; // rax

  v3 = a1;
  if ( unk_4D04964 )
  {
    v6 = a1 + 56;
    a1 = 5;
    sub_684AA0(5, 2480, v6);
    if ( a3 != 11 )
      goto LABEL_3;
  }
  else if ( a3 != 11 )
  {
LABEL_3:
    if ( a3 != 6 )
      sub_721090(a1);
    *(_BYTE *)(a2 + 176) |= 1u;
    return a2;
  }
  if ( !(unsigned int)sub_8D23B0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL)) )
  {
    v7 = sub_5C79F0(v3);
    sub_6851A0(1848, v3 + 56, v7);
    *(_BYTE *)(v3 + 8) = 0;
    return a2;
  }
  *(_BYTE *)(a2 + 192) |= 0x10u;
  return a2;
}
