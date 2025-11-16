// Function: sub_746940
// Address: 0x746940
//
void __fastcall sub_746940(char a1, __int64 a2, int a3, __int64 a4)
{
  int v6; // r15d
  int v7; // r14d
  char v8; // al
  char v9; // al

  if ( *(_BYTE *)(a4 + 137) )
    return;
  if ( (a1 & 8) != 0 )
  {
    (*(void (__fastcall **)(char *, __int64))a4)("_Atomic", a4);
    if ( (a1 & 1) == 0 )
    {
LABEL_6:
      if ( (a1 & 2) == 0 )
      {
        if ( (a1 & 4) == 0 )
        {
LABEL_8:
          if ( !(unk_4F068D0 | (*(_BYTE *)(a4 + 136) == 0)) )
          {
LABEL_13:
            v8 = 1;
            goto LABEL_14;
          }
          v6 = a1 & 0x20;
          v7 = a1 & 0x40;
          if ( (a1 & 0x10) != 0 )
          {
            (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
            goto LABEL_11;
          }
          if ( (a1 & 0x20) != 0 )
          {
LABEL_29:
            (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
LABEL_30:
            (*(void (__fastcall **)(const char *, __int64))a4)("_Nonnull", a4);
            if ( !v7 )
              goto LABEL_13;
            goto LABEL_31;
          }
          if ( (a1 & 0x40) != 0 )
          {
LABEL_31:
            (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
            goto LABEL_32;
          }
          v9 = 1;
          goto LABEL_37;
        }
        goto LABEL_27;
      }
      (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
LABEL_26:
      (*(void (__fastcall **)(char *, __int64))a4)("volatile", a4);
      if ( (a1 & 4) == 0 )
        goto LABEL_8;
LABEL_27:
      (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
      goto LABEL_28;
    }
    (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
LABEL_5:
    (*(void (__fastcall **)(char *, __int64))a4)("const", a4);
    goto LABEL_6;
  }
  if ( (a1 & 1) != 0 )
    goto LABEL_5;
  if ( (a1 & 2) != 0 )
    goto LABEL_26;
  if ( (a1 & 4) != 0 )
  {
LABEL_28:
    (*(void (__fastcall **)(const char *, __int64))a4)("__restrict__", a4);
    goto LABEL_8;
  }
  if ( !(unk_4F068D0 | (*(_BYTE *)(a4 + 136) == 0)) )
    return;
  v6 = a1 & 0x20;
  v7 = a1 & 0x40;
  if ( (a1 & 0x10) != 0 )
  {
LABEL_11:
    (*(void (__fastcall **)(const char *, __int64))a4)("_Nullable", a4);
    if ( !v6 )
    {
      if ( !v7 )
        goto LABEL_13;
      goto LABEL_31;
    }
    goto LABEL_29;
  }
  if ( (a1 & 0x20) != 0 )
    goto LABEL_30;
  v9 = 0;
  if ( (a1 & 0x40) != 0 )
  {
LABEL_32:
    (*(void (__fastcall **)(const char *, __int64))a4)("_Null_unspecified", a4);
    goto LABEL_13;
  }
LABEL_37:
  v8 = v9 & 1;
LABEL_14:
  if ( a3 )
  {
    if ( v8 )
      (*(void (__fastcall **)(char *, __int64))a4)(" ", a4);
  }
}
