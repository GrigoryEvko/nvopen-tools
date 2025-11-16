// Function: sub_640330
// Address: 0x640330
//
__int64 __fastcall sub_640330(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  _BYTE *v4; // r14
  __int64 v6; // r12
  __int64 result; // rax
  __int64 i; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  char v12; // dl
  int v13; // ecx
  __int64 v14; // rdi
  __int64 v15; // rsi
  char v16; // dl
  __int64 v17; // rsi
  char v18; // [rsp+Bh] [rbp-35h]
  unsigned __int8 v19; // [rsp+Bh] [rbp-35h]
  char v20; // [rsp+Bh] [rbp-35h]
  char v21; // [rsp+Ch] [rbp-34h]
  int v22; // [rsp+Ch] [rbp-34h]
  _BOOL4 v23; // [rsp+Ch] [rbp-34h]

  v4 = 0;
  v6 = a2;
  if ( a1 )
    v4 = *(_BYTE **)(a1 + 88);
  result = sub_8D32E0(a2);
  if ( (_DWORD)result )
  {
    if ( v4[136] != 1 )
      return sub_6854E0(252, a1);
    return result;
  }
  if ( (*(_BYTE *)(a2 + 140) & 0xFB) != 8
    || (v21 = *(_BYTE *)(a2 + 140) & 0xFB, (sub_8D4C10(a2, dword_4F077C4 != 2) & 1) == 0) )
  {
    if ( (unsigned int)sub_8D3410(a2) )
      v6 = sub_8D40F0(a2);
    while ( *(_BYTE *)(v6 + 140) == 12 )
      v6 = *(_QWORD *)(v6 + 160);
    if ( dword_4F077C4 != 2 )
    {
      result = sub_8D3B10(v6);
      if ( (_DWORD)result )
        return result;
    }
    result = sub_8D3A70(v6);
    if ( !(_DWORD)result || v4 && v4[136] == 1 )
      return result;
    if ( (*(_BYTE *)(v6 + 176) & 2) != 0 )
    {
      if ( a1 )
      {
        v14 = 5;
        v15 = 370;
        if ( dword_4F077C4 != 2 )
          return sub_6853B0(v14, v15, a1 + 48, a1);
LABEL_55:
        v15 = 369;
        v14 = qword_4D0495C == 0 ? 7 : 5;
        return sub_6853B0(v14, v15, a1 + 48, a1);
      }
    }
    else
    {
      if ( dword_4F077C4 != 2 )
        return result;
      for ( i = v6; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 178LL) & 0x20) == 0 )
      {
        result = *(_QWORD *)(v6 + 168);
        v10 = *(_QWORD **)result;
        if ( !*(_QWORD *)result )
          return result;
        while ( 1 )
        {
          v11 = v10[5];
          if ( (*(_BYTE *)(v11 + 176) & 2) != 0 )
            break;
          while ( *(_BYTE *)(v11 + 140) == 12 )
            v11 = *(_QWORD *)(v11 + 160);
          result = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
          if ( (*(_BYTE *)(result + 178) & 0x20) != 0 )
            break;
          v10 = (_QWORD *)*v10;
          if ( !v10 )
            return result;
        }
      }
      if ( a1 )
        goto LABEL_55;
    }
    if ( !a4 )
    {
      v17 = 517;
      return sub_684AC0(7, v17);
    }
LABEL_52:
    result = sub_67D3C0(516, 7, dword_4F07508);
    if ( !(_DWORD)result )
      return result;
    goto LABEL_53;
  }
  result = sub_8D3410(a2);
  v12 = v21;
  v13 = result;
  if ( (_DWORD)result )
  {
    v20 = v21;
    v23 = sub_8D23B0(a2) != 0;
    result = sub_8D40F0(a2);
    v12 = v20;
    v13 = v23;
    v6 = result;
  }
  if ( !v4 )
  {
    result = (unsigned int)sub_8D5A50(v6) | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C);
    if ( (_DWORD)result )
      return result;
    if ( (unsigned int)sub_8D3A70(v6) )
    {
      if ( !a4 )
      {
        while ( *(_BYTE *)(v6 + 140) == 12 )
          v6 = *(_QWORD *)(v6 + 160);
        return sub_685260(7, 812, dword_4F07508, v6);
      }
      result = sub_67D3C0(812, 7, dword_4F07508);
      if ( !(_DWORD)result )
        return result;
LABEL_53:
      *a4 = 1;
      return result;
    }
    if ( !a4 )
    {
      v17 = 516;
      return sub_684AC0(7, v17);
    }
    goto LABEL_52;
  }
  if ( dword_4F077C4 != 2 )
  {
    if ( (v4[88] & 0x70) == 0 )
      return sub_6854B0(257, a1);
    return result;
  }
  if ( v4[136] != 1 || (v4[172] & 8) != 0 )
  {
    v18 = v12;
    v22 = v13;
    result = sub_8DD3B0(v6);
    if ( !(_DWORD)result )
    {
      result = sub_8D5A50(v6);
      if ( !(_DWORD)result )
      {
        v16 = v18;
        if ( !dword_4F077BC || (v4[172] & 8) != 0 )
        {
          if ( dword_4F04C44 != -1 )
            goto LABEL_47;
        }
        else
        {
          if ( dword_4F04C44 != -1 )
          {
LABEL_47:
            v16 = 5;
LABEL_48:
            v19 = v16;
            if ( (unsigned int)sub_8D3A70(v6) && !(v22 | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C)) )
            {
              while ( *(_BYTE *)(v6 + 140) == 12 )
                v6 = *(_QWORD *)(v6 + 160);
              return sub_686750(v19, 811, dword_4F07508, a1, v6);
            }
            else if ( (v4[172] & 8) != 0 )
            {
              return sub_6851C0(2385, dword_4F07508);
            }
            else
            {
              return sub_685440(v19, 257, a1);
            }
          }
          if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
            goto LABEL_48;
          v16 = 5;
        }
        if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
          v16 = 5;
        goto LABEL_48;
      }
    }
  }
  return result;
}
