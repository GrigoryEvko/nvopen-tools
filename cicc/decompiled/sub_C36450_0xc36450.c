// Function: sub_C36450
// Address: 0xc36450
//
__int64 __fastcall sub_C36450(__int64 a1, char a2, int a3)
{
  int v5; // eax
  __int64 v6; // rdx
  unsigned int v7; // r14d
  int v8; // ecx
  signed int v9; // r15d
  int v10; // edx
  __int64 v11; // rax
  int v12; // ecx
  char v13; // dl
  char v14; // al
  int v15; // eax
  int v16; // edx
  unsigned int v17; // eax
  bool v18; // cc

  if ( (*(_BYTE *)(a1 + 20) & 7) == 3 || (*(_BYTE *)(a1 + 20) & 6) == 0 )
    return 0;
  v5 = sub_C34200(a1);
  v6 = *(_QWORD *)a1;
  v7 = v5 + 1;
  if ( (v5 + 1) | a3 )
  {
    v8 = *(_DWORD *)(a1 + 16);
    v9 = v7 - *(_DWORD *)(v6 + 8);
    if ( v9 + v8 > *(_DWORD *)v6 )
      return sub_C36320(a1, a2);
    if ( v9 + v8 < *(_DWORD *)(v6 + 4) )
      v9 = *(_DWORD *)(v6 + 4) - v8;
    if ( v9 < 0 )
    {
      sub_C34340(a1, -v9);
    }
    else
    {
      if ( v9 )
      {
        v15 = sub_C342A0(a1, v9);
        if ( a3 )
        {
          a3 = 1;
          if ( v15 )
          {
            v16 = 3;
            if ( v15 != 2 )
              v16 = v15;
            a3 = v16;
          }
        }
        else
        {
          a3 = v15;
        }
        v6 = *(_QWORD *)a1;
        v17 = v7 - v9;
        v18 = v7 <= v9;
        v7 = 0;
        if ( !v18 )
          v7 = v17;
      }
      if ( *(_DWORD *)(v6 + 16) == 1
        && *(_DWORD *)(v6 + 20) == 1
        && *(_DWORD *)(a1 + 16) == *(_DWORD *)v6
        && (unsigned __int8)sub_C339A0(a1) )
      {
        return sub_C36320(a1, a2);
      }
      if ( a3 )
      {
        if ( !sub_C34390(a1, a2, a3, 0) )
          goto LABEL_20;
        if ( !v7 )
          *(_DWORD *)(a1 + 16) = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
        sub_C33F10(a1);
        v10 = sub_C34200(a1);
        v7 = v10 + 1;
        v11 = *(_QWORD *)a1;
        v12 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
        if ( v10 == v12 )
        {
          if ( *(_DWORD *)(a1 + 16) != *(_DWORD *)v11 )
          {
            sub_C342A0(a1, 1u);
            return 16;
          }
          a2 = ((*(_BYTE *)(a1 + 20) & 8) != 0) + 2;
        }
        else
        {
          if ( *(_DWORD *)(v11 + 16) != 1 || *(_DWORD *)(v11 + 20) != 1 || *(_DWORD *)(a1 + 16) != *(_DWORD *)v11 )
          {
LABEL_21:
            if ( v7 == v12 )
              return 16;
            if ( !v7 )
            {
              v13 = *(_BYTE *)(a1 + 20) & 0xF8 | 3;
              *(_BYTE *)(a1 + 20) = v13;
              if ( *(_DWORD *)(v11 + 20) == 2 )
                *(_BYTE *)(a1 + 20) = v13 & 0xF7;
              if ( !*(_BYTE *)(v11 + 24) )
                sub_C35A40(a1, 0);
            }
            return 24;
          }
          if ( !(unsigned __int8)sub_C339A0(a1) )
          {
LABEL_20:
            v11 = *(_QWORD *)a1;
            v12 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
            goto LABEL_21;
          }
        }
        return sub_C36320(a1, a2);
      }
      if ( !v7 )
        goto LABEL_44;
    }
    return 0;
  }
  if ( *(_DWORD *)(v6 + 16) != 1 || *(_DWORD *)(v6 + 20) != 1 || *(_DWORD *)(a1 + 16) != *(_DWORD *)v6 )
    goto LABEL_30;
  if ( (unsigned __int8)sub_C339A0(a1) )
    return sub_C36320(a1, a2);
LABEL_44:
  v6 = *(_QWORD *)a1;
LABEL_30:
  v14 = *(_BYTE *)(a1 + 20) & 0xF8 | 3;
  *(_BYTE *)(a1 + 20) = v14;
  if ( *(_DWORD *)(v6 + 20) == 2 )
    *(_BYTE *)(a1 + 20) = v14 & 0xF7;
  if ( *(_BYTE *)(v6 + 24) )
    return 0;
  sub_C35A40(a1, 0);
  return 0;
}
