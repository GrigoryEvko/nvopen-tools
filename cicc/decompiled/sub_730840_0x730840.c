// Function: sub_730840
// Address: 0x730840
//
_BOOL8 __fastcall sub_730840(__int64 a1)
{
  __int64 *v1; // rbx
  _BOOL4 v2; // r8d
  __int64 v4; // r12
  char i; // al
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  bool v8; // al
  char v9; // al
  __int64 v10; // rdi
  char j; // al
  __int64 v12; // rax

  v1 = *(__int64 **)(a1 + 200);
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
  {
    v2 = 0;
    if ( v1 )
      return v2;
  }
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 120LL);
  for ( i = *(_BYTE *)(v4 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
    v4 = *(_QWORD *)(v4 + 160);
  if ( i != 8 || (unsigned int)sub_8D23E0(v4) )
    v6 = 1;
  else
    v6 = sub_8D4490(v4);
  if ( !v1 )
    return 1;
  while ( 1 )
  {
    v9 = *((_BYTE *)v1 + 8);
    if ( (v9 & 1) == 0 )
    {
      if ( (v9 & 2) != 0 )
        goto LABEL_17;
      v10 = *(_QWORD *)(v1[2] + 120);
      for ( j = *(_BYTE *)(v10 + 140); j == 12; j = *(_BYTE *)(v10 + 140) )
        v10 = *(_QWORD *)(v10 + 160);
      if ( j != 8 )
      {
LABEL_17:
        v8 = 0;
        v6 = 1;
        v2 = 1;
      }
      else
      {
        v12 = sub_8D4490(v10);
        v2 = 1;
        v6 = v12;
        v8 = 0;
      }
      goto LABEL_13;
    }
    v7 = v1[2];
    if ( v7 < 0 )
      return 0;
    v2 = v7 < v6;
    v8 = v7 >= v6;
LABEL_13:
    v1 = (__int64 *)*v1;
    if ( !v1 || v8 )
      return v2;
  }
}
