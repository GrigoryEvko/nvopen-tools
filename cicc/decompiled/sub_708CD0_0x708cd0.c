// Function: sub_708CD0
// Address: 0x708cd0
//
__int64 __fastcall sub_708CD0(__int64 a1, unsigned __int8 a2)
{
  __int64 v2; // rax
  bool v3; // sf
  __int64 v5; // rax

  sub_7E99A0(a1, a2 == 7);
  if ( a2 == 7 )
  {
    *(_BYTE *)(a1 + 136) = 0;
    v5 = sub_8D07C0();
    *(_BYTE *)(v5 + 20) = 7;
    *(_QWORD *)v5 = a1;
    if ( (*(_BYTE *)(a1 - 8) & 2) == 0 )
      *(_QWORD *)(v5 + 8) = a1;
    *(_QWORD *)(a1 + 32) = v5;
    return sub_7604D0(a1, 7);
  }
  else
  {
    *(_BYTE *)(a1 + 172) = 0;
    v2 = sub_8D07C0();
    *(_BYTE *)(v2 + 20) = a2;
    *(_QWORD *)v2 = a1;
    if ( (*(_BYTE *)(a1 - 8) & 2) == 0 )
      *(_QWORD *)(v2 + 8) = a1;
    v3 = *(char *)(a1 + 192) < 0;
    *(_QWORD *)(a1 + 32) = v2;
    if ( v3 && dword_4F068EC && (*(_BYTE *)(a1 + 193) & 4) == 0 && *(char *)(a1 + 203) >= 0 )
      sub_89A080(a1);
    return sub_7604D0(a1, a2);
  }
}
