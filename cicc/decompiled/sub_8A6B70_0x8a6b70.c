// Function: sub_8A6B70
// Address: 0x8a6b70
//
void __fastcall sub_8A6B70(unsigned __int64 a1, __int64 a2)
{
  __int64 v3; // r13
  _QWORD *i; // rbx
  __int64 v5; // r15
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  int v8; // edx
  int v9; // eax
  _QWORD *j; // rbx
  unsigned __int8 v11; // [rsp+Fh] [rbp-31h]

  if ( !a2 )
    a2 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL);
  v3 = *(_QWORD *)(a2 + 88);
  for ( i = *(_QWORD **)(v3 + 168); i; i = (_QWORD *)*i )
  {
    while ( 1 )
    {
      v5 = i[1];
      v6 = *(_QWORD *)(v5 + 88);
      if ( (*(_BYTE *)(v6 + 177) & 0x20) == 0
        && (*(_BYTE *)(v6 + 178) & 1) == 0
        && !(unsigned int)sub_8D23B0(v6)
        && sub_8A64A0(a1, v5, 0, 0) )
      {
        break;
      }
LABEL_12:
      i = (_QWORD *)*i;
      if ( !i )
        goto LABEL_13;
    }
    v7 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 72LL);
    v8 = dword_4F077BC == 0 ? 8 : 5;
    if ( *(_QWORD *)(*(_QWORD *)(v7 + 88) + 152LL) )
    {
      v11 = dword_4F077BC == 0 ? 8 : 5;
      v9 = sub_8A6950(a1, v7);
      LOBYTE(v8) = v11;
      if ( v9 <= 0 )
      {
        if ( !v9 )
          sub_6853B0(v11, 0x34Eu, (FILE *)(a1 + 48), v5);
        goto LABEL_12;
      }
    }
    sub_6853B0(v8, 0x34Du, (FILE *)(a1 + 48), v5);
  }
LABEL_13:
  for ( j = *(_QWORD **)(v3 + 96); j; j = (_QWORD *)*j )
    sub_8A6B70(a1, j[1]);
}
