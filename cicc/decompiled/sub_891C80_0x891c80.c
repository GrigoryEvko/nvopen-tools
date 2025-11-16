// Function: sub_891C80
// Address: 0x891c80
//
_BOOL8 __fastcall sub_891C80(__int64 a1, int a2, int a3, char a4)
{
  char v6; // al
  __int64 v7; // rdx
  _BOOL8 result; // rax
  _BYTE *v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rax
  int v13; // eax
  int v14; // eax
  int v15; // r8d
  char v16; // [rsp+Ch] [rbp-34h]
  char v17; // [rsp+Ch] [rbp-34h]
  char v18; // [rsp+Ch] [rbp-34h]
  char v19; // [rsp+Ch] [rbp-34h]
  char v20; // [rsp+Ch] [rbp-34h]

  v6 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v6 - 10) <= 1u )
  {
    v9 = *(_BYTE **)(a1 + 88);
    v10 = *(_QWORD *)(a1 + 96);
    if ( (v9[193] & 0x10) != 0 )
    {
      if ( a2 && (v9[194] & 0x40) == 0 )
      {
        v17 = a4;
        sub_685440(7 - ((a3 == 0) - 1), 0x1E6u, a1);
        a4 = v17;
        result = 0;
        goto LABEL_12;
      }
    }
    else if ( v10 )
    {
      if ( (v9[195] & 2) != 0 )
      {
        if ( a4 != 16 && a3 != 0 && (a2 & 1) != 0 )
        {
          v16 = a4;
          sub_685440(unk_4F07471, 0x1EAu, a1);
          a4 = v16;
          result = 0;
          goto LABEL_12;
        }
      }
      else if ( (v9[206] & 0x10) == 0 )
      {
        v11 = *(_QWORD *)v9;
        if ( (*(_BYTE *)(*(_QWORD *)v9 + 104LL) & 1) != 0 )
        {
          v20 = a4;
          v13 = sub_8796F0(v11);
          a4 = v20;
        }
        else
        {
          v12 = *(_QWORD *)(v11 + 88);
          if ( *(_BYTE *)(v11 + 80) == 20 )
            v12 = *(_QWORD *)(v12 + 176);
          v13 = (*(_BYTE *)(v12 + 208) & 4) != 0;
        }
        if ( !v13 )
        {
          v19 = a4;
          v14 = sub_8919F0(v10, 0);
          a4 = v19;
          v15 = v14;
          result = 1;
          if ( v15 )
          {
            result = a3 == 0;
            if ( a3 )
            {
              if ( (a2 & 1) != 0 )
              {
                sub_685440(8u, 0x1E7u, a1);
                a4 = v19;
                result = a3 == 0;
              }
            }
          }
          goto LABEL_12;
        }
      }
    }
    else if ( a2 )
    {
      v18 = a4;
      sub_6854E0(0x1E5u, a1);
      a4 = v18;
      result = 0;
LABEL_12:
      if ( !unk_4D04210 || v9[172] != 2 || a4 != 16 || a3 )
        return result;
      if ( a2 )
        sub_6854E0(0x640u, *(_QWORD *)(v10 + 32));
      return 0;
    }
    result = 0;
    goto LABEL_12;
  }
  if ( v6 == 9 || v6 == 7 )
  {
    v7 = *(_QWORD *)(a1 + 88);
  }
  else
  {
    if ( v6 != 21 )
      BUG();
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 192LL);
  }
  result = 1;
  if ( *(char *)(v7 + 170) < 0 )
  {
    if ( a4 != 16 && a3 != 0 && (a2 & 1) != 0 )
    {
      sub_685440(unk_4F07471, 0x1EAu, a1);
      return 0;
    }
    return 0;
  }
  return result;
}
