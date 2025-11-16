// Function: sub_394AFB0
// Address: 0x394afb0
//
char __fastcall sub_394AFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char result; // al
  unsigned __int8 *v5; // r13
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned int v9; // edx
  int v10; // esi
  unsigned int v11; // r15d
  unsigned int v12; // r12d
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // edx

  if ( !*(_BYTE *)(a1 + 16) )
    return 1;
  if ( (*(_BYTE *)(a1 + 33) & 0x1C) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 32) & 0xF) == 0xA )
      return 16;
    if ( sub_394AEC0(*(_QWORD *)(a1 - 24), a2, a3, a4) )
    {
      if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
      {
        if ( (*(_BYTE *)(a1 + 34) & 0x20) == 0 && *(char *)(a2 + 792) >= 0 )
        {
          result = 14;
          if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
            return (*(_BYTE *)(a1 + 32) & 0xF) == 0 ? 15 : 13;
          return result;
        }
        return 17;
      }
    }
    else if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
    {
      return 17;
    }
    v5 = *(unsigned __int8 **)(a1 - 24);
    if ( (unsigned __int8)sub_1593ED0((__int64)v5) )
    {
      v6 = sub_1700490(a2);
      if ( (unsigned int)(v6 - 3) > 2 && v6 )
        return 18;
      return 3;
    }
    if ( *(_BYTE *)(a1 + 32) >> 6 != 2 )
      return 3;
    v7 = *(_QWORD *)v5;
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 14 )
    {
      v8 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)(v8 + 8) == 11 )
      {
        v9 = *(_DWORD *)(v8 + 8);
        if ( (((v9 >> 8) - 8) & 0xFFFFFFF7) == 0 || v9 >> 8 == 32 )
        {
          v10 = v5[16];
          if ( (unsigned int)(v10 - 11) > 1 )
          {
            if ( (_BYTE)v10 == 10 && *(_QWORD *)(v7 + 32) == 1 )
              goto LABEL_46;
          }
          else
          {
            v11 = sub_15958F0((__int64)v5) - 1;
            if ( !sub_1595A50((__int64)v5, v11) )
            {
              v12 = 0;
              if ( !v11 )
              {
LABEL_48:
                v9 = *(_DWORD *)(v8 + 8);
LABEL_46:
                v16 = v9 >> 8;
                if ( v16 == 8 )
                  return 4;
                else
                  return (v16 != 16) + 5;
              }
              while ( sub_1595A50((__int64)v5, v12) )
              {
                if ( v11 == ++v12 )
                  goto LABEL_48;
              }
            }
          }
        }
      }
    }
    v13 = sub_1632FA0(*(_QWORD *)(a1 + 40));
    v14 = sub_12BE0A0(v13, *(_QWORD *)v5);
    v15 = v14;
    if ( v14 == 16 )
      return 9;
    if ( v14 > 0x10 )
    {
      result = 10;
      if ( v15 == 32 )
        return result;
    }
    else
    {
      result = 7;
      if ( v15 == 4 )
        return result;
      if ( v15 == 8 )
        return 8;
    }
    return 3;
  }
  if ( !sub_394AEC0(*(_QWORD *)(a1 - 24), a2, a3, a4) )
    return 12;
  if ( (*(_BYTE *)(a1 + 80) & 1) != 0 )
    return 12;
  if ( (*(_BYTE *)(a1 + 34) & 0x20) != 0 )
    return 12;
  result = 11;
  if ( *(char *)(a2 + 792) < 0 )
    return 12;
  return result;
}
