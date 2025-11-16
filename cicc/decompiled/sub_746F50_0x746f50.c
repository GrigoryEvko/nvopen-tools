// Function: sub_746F50
// Address: 0x746f50
//
__int64 __fastcall sub_746F50(int a1, __int64 a2)
{
  char v3; // al
  int v4; // esi
  unsigned int v5; // r12d
  char *p_s; // rax
  void (__fastcall *v7)(char *, __int64); // rax
  char v9; // al
  __int64 v10; // rcx
  char s; // [rsp+16h] [rbp-2Ah] BYREF
  _BYTE v12[41]; // [rsp+17h] [rbp-29h] BYREF

  if ( isprint((unsigned __int8)a1) )
  {
    v3 = *(_BYTE *)(a2 + 136);
    if ( v3 && (a1 & 0x80u) != 0 )
    {
      v4 = (char)a1;
    }
    else if ( !unk_4F068E4 || (v4 = 39, (_BYTE)a1 != 39) )
    {
      if ( (unsigned __int8)(a1 - 34) <= 0x3Au )
      {
        v10 = 0x400000000000021LL;
        if ( _bittest64(&v10, (unsigned int)(a1 - 34)) || (_BYTE)a1 == 63 && v3 && !*(_BYTE *)(a2 + 137) )
        {
          s = 92;
          v5 = 2;
          p_s = v12;
          goto LABEL_8;
        }
      }
LABEL_7:
      v5 = 1;
      p_s = &s;
LABEL_8:
      *(_WORD *)p_s = (unsigned __int8)a1;
      goto LABEL_9;
    }
  }
  else
  {
    if ( (_BYTE)a1 == 9 )
    {
      v9 = 116;
      if ( !*(_BYTE *)(a2 + 144) )
      {
LABEL_15:
        v12[0] = v9;
        v7 = *(void (__fastcall **)(char *, __int64))(a2 + 8);
        v5 = 2;
        s = 92;
        v12[1] = 0;
        if ( v7 )
          goto LABEL_10;
LABEL_16:
        (*(void (__fastcall **)(char *, __int64))a2)(&s, a2);
        return v5;
      }
      goto LABEL_7;
    }
    v4 = (char)a1;
    switch ( (char)a1 )
    {
      case 7:
        if ( (!*(_BYTE *)(a2 + 136) || !*(_BYTE *)(a2 + 141)) && !*(_BYTE *)(a2 + 137) )
        {
          v9 = 97;
          goto LABEL_15;
        }
        v4 = 7;
        break;
      case 8:
        v9 = 98;
        goto LABEL_15;
      case 10:
        v9 = 110;
        goto LABEL_15;
      case 11:
        v9 = 118;
        goto LABEL_15;
      case 12:
        v9 = 102;
        goto LABEL_15;
      case 13:
        v9 = 114;
        goto LABEL_15;
      default:
        break;
    }
  }
  v5 = 4;
  sprintf(&s, "\\%03o", v4 & ((1 << unk_4F06B9C) - 1));
LABEL_9:
  v7 = *(void (__fastcall **)(char *, __int64))(a2 + 8);
  if ( !v7 )
    goto LABEL_16;
LABEL_10:
  v7(&s, a2);
  return v5;
}
