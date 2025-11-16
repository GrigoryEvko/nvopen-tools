// Function: sub_13CC800
// Address: 0x13cc800
//
__int64 *__fastcall sub_13CC800(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // r13
  unsigned __int8 v7; // al
  unsigned __int8 v8; // dl
  __int64 v10; // r13
  __int64 *v11; // r15
  unsigned __int8 v12; // al
  unsigned __int8 v13; // dl
  __int64 v14; // r13
  unsigned __int8 v15; // al
  unsigned __int8 v16; // dl
  __int64 v17; // r13
  unsigned __int8 v18; // al
  unsigned __int8 v19; // dl

  v2 = a1;
  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  if ( v3 > 0 )
  {
    v5 = &a1[4 * v3];
    while ( 1 )
    {
      v6 = *v2;
      v7 = *(_BYTE *)(*v2 + 16);
      if ( v7 > 0x17u )
      {
        if ( v7 != 53 || !*(_QWORD *)(v6 + 40) || !sub_15F2060(*v2) || !(unsigned __int8)sub_15F8F00(v6) )
          return v2;
      }
      else if ( v7 > 3u )
      {
        if ( v7 != 17 || !(unsigned __int8)sub_15E0450(*v2) )
          return v2;
      }
      else
      {
        v8 = *(_BYTE *)(v6 + 32);
        if ( (v8 & 0xFu) - 7 > 1 && (((v8 & 0x30) - 16) & 0xE0) != 0 && v8 >> 6 != 2
          || (*(_BYTE *)(v6 + 33) & 0x1C) != 0 )
        {
          return v2;
        }
      }
      v10 = v2[1];
      v11 = v2 + 1;
      v12 = *(_BYTE *)(v10 + 16);
      if ( v12 > 0x17u )
      {
        if ( v12 != 53 || !*(_QWORD *)(v10 + 40) || !sub_15F2060(v2[1]) || !(unsigned __int8)sub_15F8F00(v10) )
          return v11;
      }
      else if ( v12 > 3u )
      {
        if ( v12 != 17 || !(unsigned __int8)sub_15E0450(v2[1]) )
          return v11;
      }
      else
      {
        v13 = *(_BYTE *)(v10 + 32);
        if ( (v13 & 0xFu) - 7 > 1 && (((v13 & 0x30) - 16) & 0xE0) != 0 && v13 >> 6 != 2 )
          return ++v2;
        if ( (*(_BYTE *)(v10 + 33) & 0x1C) != 0 )
          return v11;
      }
      v14 = v2[2];
      v11 = v2 + 2;
      v15 = *(_BYTE *)(v14 + 16);
      if ( v15 > 0x17u )
      {
        if ( v15 != 53 || !*(_QWORD *)(v14 + 40) || !sub_15F2060(v2[2]) || !(unsigned __int8)sub_15F8F00(v14) )
          return v11;
      }
      else if ( v15 > 3u )
      {
        if ( v15 != 17 || !(unsigned __int8)sub_15E0450(v2[2]) )
          return v11;
      }
      else
      {
        v16 = *(_BYTE *)(v14 + 32);
        if ( (v16 & 0xFu) - 7 > 1 && (((v16 & 0x30) - 16) & 0xE0) != 0 && v16 >> 6 != 2 )
        {
          v2 += 2;
          return v2;
        }
        if ( (*(_BYTE *)(v14 + 33) & 0x1C) != 0 )
          return v11;
      }
      v17 = v2[3];
      v11 = v2 + 3;
      v18 = *(_BYTE *)(v17 + 16);
      if ( v18 > 0x17u )
      {
        if ( v18 != 53 || !*(_QWORD *)(v17 + 40) || !sub_15F2060(v2[3]) )
          return v11;
        if ( !(unsigned __int8)sub_15F8F00(v17) )
        {
          v2 += 3;
          return v2;
        }
      }
      else if ( v18 > 3u )
      {
        if ( v18 != 17 || !(unsigned __int8)sub_15E0450(v2[3]) )
          return v11;
      }
      else
      {
        v19 = *(_BYTE *)(v17 + 32);
        if ( (v19 & 0xFu) - 7 > 1 && (((v19 & 0x30) - 16) & 0xE0) != 0 && v19 >> 6 != 2 )
        {
          v2 += 3;
          return v2;
        }
        if ( (*(_BYTE *)(v17 + 33) & 0x1C) != 0 )
          return v11;
      }
      v2 += 4;
      if ( v2 == v5 )
      {
        v4 = (a2 - (__int64)v2) >> 3;
        break;
      }
    }
  }
  if ( v4 == 2 )
    goto LABEL_72;
  if ( v4 == 3 )
  {
    if ( !sub_13CBD60(*v2) )
      return v2;
    ++v2;
LABEL_72:
    if ( !sub_13CBD60(*v2) )
      return v2;
    ++v2;
    goto LABEL_74;
  }
  if ( v4 != 1 )
    return (__int64 *)a2;
LABEL_74:
  if ( sub_13CBD60(*v2) )
    return (__int64 *)a2;
  return v2;
}
