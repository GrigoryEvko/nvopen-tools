// Function: sub_5C9CC0
// Address: 0x5c9cc0
//
__int64 __fastcall sub_5C9CC0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  _QWORD *v4; // r14
  _QWORD *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int16 v8; // [rsp+Eh] [rbp-32h]

  if ( unk_4F06418 <= 0x1Cu )
  {
    if ( unk_4F06418 > 8u )
    {
      switch ( unk_4F06418 )
      {
        case 9:
        case 0x1A:
        case 0x1C:
          return 0;
        case 0x19:
          v8 = 26;
          goto LABEL_9;
        case 0x1B:
          v6 = sub_7276D0();
          *(_BYTE *)(v6 + 10) = 1;
          v1 = v6;
          *(_QWORD *)(v6 + 24) = unk_4F063F8;
          *(_QWORD *)(v6 + 32) = unk_4F063F0;
          *(_WORD *)(v6 + 8) = unk_4F06418;
          *(_QWORD *)(v6 + 40) = sub_7C9D40();
          sub_7B8B50();
          v8 = 28;
          goto LABEL_10;
        default:
          goto LABEL_16;
      }
    }
    goto LABEL_16;
  }
  if ( unk_4F06418 == 73 )
  {
    v8 = 74;
LABEL_9:
    v3 = sub_7276D0();
    *(_BYTE *)(v3 + 10) = 1;
    v1 = v3;
    *(_QWORD *)(v3 + 24) = unk_4F063F8;
    *(_QWORD *)(v3 + 32) = unk_4F063F0;
    *(_WORD *)(v3 + 8) = unk_4F06418;
    *(_QWORD *)(v3 + 40) = sub_7C9D40();
    sub_7B8B50();
LABEL_10:
    v4 = (_QWORD *)v1;
    v5 = (_QWORD *)sub_5C9CC0(a1);
    for ( *(_QWORD *)v1 = v5; v5; *v4 = v5 )
    {
      do
      {
        v4 = v5;
        v5 = (_QWORD *)*v5;
      }
      while ( v5 );
      v5 = (_QWORD *)sub_5C9CC0(a1);
    }
    if ( unk_4F06418 == v8 )
    {
      v7 = sub_7276D0();
      *(_BYTE *)(v7 + 10) = 1;
      *(_QWORD *)(v7 + 24) = unk_4F063F8;
      *(_QWORD *)(v7 + 32) = unk_4F063F0;
      *(_WORD *)(v7 + 8) = unk_4F06418;
      *(_QWORD *)(v7 + 40) = sub_7C9D40();
      sub_7B8B50();
      *v4 = v7;
    }
    else if ( !*a1 )
    {
      *a1 = v1;
    }
  }
  else
  {
    if ( unk_4F06418 != 74 )
    {
LABEL_16:
      v1 = sub_7276D0();
      *(_BYTE *)(v1 + 10) = 1;
      *(_QWORD *)(v1 + 24) = unk_4F063F8;
      *(_QWORD *)(v1 + 32) = unk_4F063F0;
      *(_WORD *)(v1 + 8) = unk_4F06418;
      *(_QWORD *)(v1 + 40) = sub_7C9D40();
      sub_7B8B50();
      return v1;
    }
    return 0;
  }
  return v1;
}
