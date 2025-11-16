// Function: sub_1698EC0
// Address: 0x1698ec0
//
__int64 __fastcall sub_1698EC0(__int16 **a1, unsigned int a2, unsigned int a3)
{
  int v5; // eax
  unsigned int v6; // ebx
  int v7; // ecx
  unsigned int v8; // r15d
  signed int v9; // eax
  int v10; // eax
  int v11; // ebx
  int v12; // edx
  __int64 v13; // rdx
  unsigned int v14; // eax

  if ( (*((_BYTE *)a1 + 18) & 7) == 3 || (*((_BYTE *)a1 + 18) & 6) == 0 )
    return 0;
  v5 = sub_1698BA0((__int64)a1);
  v6 = v5 + 1;
  if ( v5 == -1 )
  {
    v13 = a3;
    if ( a3 )
      goto LABEL_17;
    goto LABEL_31;
  }
  v7 = *((__int16 *)a1 + 8);
  v8 = v6 - *((_DWORD *)*a1 + 1);
  if ( (int)(v8 + v7) > **a1 )
    return sub_1698D70((__int64)a1, a2);
  v9 = (*a1)[1];
  if ( (int)(v8 + v7) < v9 )
    v8 = v9 - v7;
  if ( (v8 & 0x80000000) != 0 )
  {
    sub_1698CA0((__int64)a1, -v8);
    return 0;
  }
  if ( !v8 )
  {
    if ( a3 )
    {
      if ( !sub_1698E10((__int64)a1, a2, a3, 0) )
      {
        if ( v6 != *((_DWORD *)*a1 + 1) )
          return 24;
        return 16;
      }
      goto LABEL_12;
    }
    return 0;
  }
  v14 = sub_1698C00((__int64)a1, v8);
  if ( !a3 )
  {
    if ( v6 > v8 )
    {
      v11 = v6 - v8;
      if ( v14 )
        goto LABEL_24;
      if ( v11 )
        return 0;
    }
    else if ( v14 )
    {
      goto LABEL_38;
    }
LABEL_31:
    *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF8 | 3;
    return 0;
  }
  if ( !v14 )
  {
    v14 = 1;
    if ( v6 > v8 )
    {
      v11 = v6 - v8;
      v14 = 1;
      goto LABEL_24;
    }
    goto LABEL_38;
  }
  if ( v14 == 2 )
  {
    v14 = 3;
    if ( v6 > v8 )
      goto LABEL_42;
  }
  else if ( v6 > v8 )
  {
LABEL_42:
    v11 = v6 - v8;
LABEL_24:
    if ( !sub_1698E10((__int64)a1, a2, v14, 0) )
    {
      v12 = *((_DWORD *)*a1 + 1);
LABEL_13:
      if ( v11 != v12 )
      {
        if ( v11 )
          return 24;
        goto LABEL_19;
      }
      return 16;
    }
    if ( v11 )
      goto LABEL_12;
    goto LABEL_26;
  }
LABEL_38:
  v13 = v14;
LABEL_17:
  if ( !sub_1698E10((__int64)a1, a2, v13, 0) )
  {
    if ( *((_DWORD *)*a1 + 1) )
    {
LABEL_19:
      *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF8 | 3;
      return 24;
    }
    return 16;
  }
LABEL_26:
  *((_WORD *)a1 + 8) = (*a1)[1];
LABEL_12:
  sub_16988A0((__int64)a1);
  v10 = sub_1698BA0((__int64)a1);
  v11 = v10 + 1;
  v12 = *((_DWORD *)*a1 + 1);
  if ( v10 != v12 )
    goto LABEL_13;
  if ( *((_WORD *)a1 + 8) == **a1 )
  {
    *((_BYTE *)a1 + 18) &= 0xF8u;
    return 20;
  }
  else
  {
    sub_1698C00((__int64)a1, 1u);
    return 16;
  }
}
