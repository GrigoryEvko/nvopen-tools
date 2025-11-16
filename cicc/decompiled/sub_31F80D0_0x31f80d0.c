// Function: sub_31F80D0
// Address: 0x31f80d0
//
__int64 __fastcall sub_31F80D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r13d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _DWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _DWORD *v22; // rax
  __int64 v23; // rdx

  v2 = *(_QWORD *)(a2 + 24) >> 3;
  switch ( *(_BYTE *)(a2 + 44) )
  {
    case 2:
      v10 = (unsigned int)(v2 - 1);
      if ( (unsigned int)v10 > 0xF )
        return 0;
      v3 = dword_44D4F00[v10];
      goto LABEL_7;
    case 3:
      v11 = (unsigned int)(v2 - 4);
      if ( (unsigned int)v11 > 0x1C )
        return 0;
      v3 = dword_44D4E80[v11];
      goto LABEL_7;
    case 4:
      v12 = (unsigned int)(v2 - 2);
      if ( (unsigned int)v12 > 0xE )
        return 0;
      v3 = dword_44D4E40[v12];
      goto LABEL_7;
    case 5:
      switch ( (int)v2 )
      {
        case 1:
          goto LABEL_23;
        case 2:
          return 17;
        case 4:
          v20 = (_QWORD *)sub_A547D0(a2, 2);
          if ( v21 == 8 && *v20 == 0x746E6920676E6F6CLL )
            return 18;
          v22 = (_DWORD *)sub_A547D0(a2, 2);
          if ( v23 == 4 && *v22 == 1735290732 )
            return 18;
          else
            return 116;
        case 8:
          return 19;
        case 16:
          return 20;
        default:
          return 0;
      }
    case 6:
      if ( (_DWORD)v2 != 1 )
        return 0;
LABEL_23:
      v3 = 16;
      goto LABEL_24;
    case 7:
      v15 = (unsigned int)(v2 - 1);
      if ( (unsigned int)v15 > 0xF )
        return 0;
      v3 = dword_44D4E00[v15];
      goto LABEL_7;
    case 8:
      if ( (_DWORD)v2 != 1 )
        return 0;
      v3 = 32;
      goto LABEL_24;
    case 0x10:
      v5 = (unsigned int)(v2 - 1);
      if ( (unsigned int)v5 > 3 )
        return 0;
      v3 = *(_DWORD *)&asc_44D4DF0[4 * v5];
LABEL_7:
      if ( v3 == 117 )
      {
        v16 = sub_A547D0(a2, 2);
        if ( v17 == 17
          && !(*(_QWORD *)v16 ^ 0x736E7520676E6F6CLL | *(_QWORD *)(v16 + 8) ^ 0x6E692064656E6769LL)
          && *(_BYTE *)(v16 + 16) == 116 )
        {
          return 34;
        }
        v18 = sub_A547D0(a2, 2);
        if ( v19 == 13
          && *(_QWORD *)v18 == 0x64656E6769736E75LL
          && *(_DWORD *)(v18 + 8) == 1852795936
          && *(_BYTE *)(v18 + 12) == 103 )
        {
          return 34;
        }
        return v3;
      }
      if ( v3 == 33 )
      {
        v6 = sub_A547D0(a2, 2);
        if ( v7 == 7 && *(_DWORD *)v6 == 1634231159 && *(_WORD *)(v6 + 4) == 24434 && *(_BYTE *)(v6 + 6) == 116 )
          return 113;
        v8 = sub_A547D0(a2, 2);
        if ( v9 == 9 && *(_QWORD *)v8 == 0x5F72616863775F5FLL && *(_BYTE *)(v8 + 8) == 116 )
          return 113;
        return v3;
      }
      if ( ((v3 - 16) & 0xFFFFFFEF) != 0 )
        return v3;
LABEL_24:
      v13 = (_DWORD *)sub_A547D0(a2, 2);
      if ( v14 != 4 )
        return v3;
      if ( *v13 == 1918986339 )
        return 112;
      return v3;
    default:
      return 0;
  }
}
