// Function: sub_875C60
// Address: 0x875c60
//
__int64 __fastcall sub_875C60(__int64 a1, int a2, FILE *a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  char v5; // al
  char v7; // r15
  unsigned int v8; // r14d
  _BOOL4 v9; // ebx
  unsigned int v10; // r12d
  _BOOL4 v11; // r14d

  result = 1;
  v4 = *(_QWORD *)(a1 + 88);
  if ( (*(_BYTE *)(v4 + 206) & 0x10) != 0 )
  {
    v5 = *(_BYTE *)(v4 + 174);
    if ( a2 )
    {
      v7 = 8;
      if ( !(dword_4F077BC | (unsigned int)qword_4F077B4) )
        v7 = unk_4F07471;
      if ( v5 != 1 )
      {
        v8 = 1815;
        goto LABEL_7;
      }
    }
    else
    {
      v7 = 8;
      if ( v5 != 1 )
      {
        v8 = 1776;
        if ( a3 )
          goto LABEL_8;
        return !sub_67D3C0((int *)v8, v7, dword_4F07508);
      }
    }
    if ( (unsigned int)sub_72F310(*(_QWORD *)(a1 + 88), 0) )
    {
      v10 = (*(_BYTE *)(v4 + 194) & 0x40) == 0 ? 1790 : 3093;
    }
    else
    {
      if ( (*(_BYTE *)(v4 + 194) & 0x40) == 0 )
      {
        v8 = a2 == 0 ? 1776 : 1815;
LABEL_7:
        if ( a3 )
        {
LABEL_8:
          v9 = sub_67D370((int *)v8, v7, a3);
          sub_6853B0(v7, v8, a3, a1);
          return !v9;
        }
        return !sub_67D3C0((int *)v8, v7, dword_4F07508);
      }
      v10 = 3093;
    }
    if ( a3 )
    {
      v11 = sub_67D370((int *)v10, v7, a3);
      sub_685260(v7, v10, a3, *(_QWORD *)(*(_QWORD *)(v4 + 40) + 32LL));
      return !v11;
    }
    else
    {
      return !sub_67D3C0((int *)v10, v7, dword_4F07508);
    }
  }
  return result;
}
