// Function: sub_31578C0
// Address: 0x31578c0
//
__int64 __fastcall sub_31578C0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int16 v4; // ax
  char v5; // al
  unsigned __int8 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // edx
  int v10; // esi
  unsigned int v11; // r15d
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  char v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  bool v23; // al
  unsigned int v24; // edx
  _QWORD v25[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)a1 && *(_BYTE *)a1 != 23 )
  {
    if ( (*(_BYTE *)(a1 + 33) & 0x1C) != 0 )
    {
      v5 = sub_31574E0(*(unsigned __int8 **)(a1 - 32));
      v2 = 13;
      if ( v5 && (*(_BYTE *)(a1 + 80) & 1) == 0 && (*(_BYTE *)(a1 + 35) & 4) == 0 && (*(_BYTE *)(a2 + 865) & 1) == 0 )
        return (unsigned __int8)((*(_BYTE *)(a1 + 32) & 0xFu) - 7 < 2 ? 14 : 12);
      return v2;
    }
    v2 = 18;
    if ( (*(_BYTE *)(a1 + 32) & 0xF) == 0xA )
      return v2;
    if ( sub_31574E0(*(unsigned __int8 **)(a1 - 32)) )
    {
      v4 = (*(_WORD *)(a1 + 34) >> 1) & 0x200;
      if ( (*(_BYTE *)(a1 + 80) & 1) != 0 )
      {
        if ( !v4 )
          goto LABEL_19;
      }
      else if ( !v4 )
      {
        v2 = 19;
        if ( (*(_BYTE *)(a2 + 865) & 1) == 0 )
        {
          v2 = 16;
          if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
            return (unsigned __int8)((*(_BYTE *)(a1 + 32) & 0xF) == 0 ? 17 : 15);
        }
        return v2;
      }
    }
    else if ( (*(_BYTE *)(a1 + 35) & 4) == 0 )
    {
      goto LABEL_41;
    }
    if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    {
      v20 = sub_B91C10(a1, 33);
      if ( v20 )
      {
        v21 = (*(_BYTE *)(v20 - 16) & 2) != 0 ? *(_DWORD *)(v20 - 24) : (*(_WORD *)(v20 - 16) >> 6) & 0xF;
        v2 = 1;
        if ( !v21 )
          return v2;
      }
    }
LABEL_41:
    v2 = 19;
    if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
      return v2;
LABEL_19:
    v6 = *(unsigned __int8 **)(a1 - 32);
    if ( sub_AC2F30(v6) )
    {
      v22 = sub_23CF1A0(a2);
      if ( (unsigned int)(v22 - 3) <= 2 )
        return 4;
      if ( !v22 )
        return 4;
      v23 = sub_AC2F10(v6);
      v2 = 20;
      if ( !v23 )
        return 4;
      return v2;
    }
    v2 = 4;
    if ( *(_BYTE *)(a1 + 32) >> 6 != 2 )
      return v2;
    v7 = *((_QWORD *)v6 + 1);
    if ( *(_BYTE *)(v7 + 8) == 16 )
    {
      v8 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)(v8 + 8) == 12 )
      {
        v9 = *(_DWORD *)(v8 + 8);
        if ( (((v9 >> 8) - 8) & 0xFFFFFFF7) == 0 || v9 >> 8 == 32 )
        {
          v10 = *v6;
          if ( (unsigned int)(v10 - 15) > 1 )
          {
            if ( (_BYTE)v10 == 14 && *(_QWORD *)(v7 + 32) == 1 )
              goto LABEL_54;
          }
          else
          {
            v11 = sub_AC5290((__int64)v6) - 1;
            if ( !sub_AC5320((__int64)v6, v11) )
            {
              v12 = 0;
              if ( !v11 )
              {
LABEL_56:
                v9 = *(_DWORD *)(v8 + 8);
LABEL_54:
                v24 = v9 >> 8;
                if ( v24 == 8 )
                  return 5;
                else
                  return (unsigned __int8)((v24 != 16) + 6);
              }
              while ( sub_AC5320((__int64)v6, v12) )
              {
                if ( v11 == ++v12 )
                  goto LABEL_56;
              }
            }
          }
        }
      }
    }
    v13 = sub_B2F730(a1);
    v14 = *((_QWORD *)v6 + 1);
    v15 = v13;
    v16 = sub_AE5020(v13, v14);
    v17 = sub_9208B0(v15, v14);
    v25[1] = v18;
    v25[0] = (((unsigned __int64)(v17 + 7) >> 3) + (1LL << v16) - 1) >> v16 << v16;
    v19 = sub_CA1930(v25);
    if ( v19 == 16 )
    {
      return 10;
    }
    else if ( v19 > 0x10 )
    {
      v2 = 4;
      if ( v19 == 32 )
        return 11;
    }
    else
    {
      v2 = 8;
      if ( v19 != 4 )
        return (unsigned __int8)(5 * (v19 == 8) + 4);
    }
    return v2;
  }
  return 2;
}
