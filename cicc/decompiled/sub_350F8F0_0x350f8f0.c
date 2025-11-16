// Function: sub_350F8F0
// Address: 0x350f8f0
//
__int64 __fastcall sub_350F8F0(__int64 a1, unsigned __int64 a2)
{
  char v2; // r8
  unsigned __int64 v3; // rdi
  __int64 v4; // rcx
  int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int16 v7; // dx
  unsigned __int8 v8; // di
  unsigned __int64 v9; // rsi
  unsigned int v11; // r8d

  v2 = a2 & 2;
  v3 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  if ( (a2 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
  {
    if ( (a2 & 1) != 0 )
    {
LABEL_12:
      v4 = 0;
      goto LABEL_13;
    }
    v5 = (unsigned __int16)((unsigned int)a2 >> 8);
    goto LABEL_8;
  }
  if ( (a2 & 6) != 2 )
  {
    if ( (a2 & 1) != 0 )
    {
      if ( v2 )
      {
        v4 = 0;
LABEL_24:
        v6 = HIWORD(a2);
        goto LABEL_14;
      }
      goto LABEL_12;
    }
    v5 = (unsigned __int16)((unsigned int)a2 >> 8);
    if ( v2 )
    {
      v6 = (unsigned __int16)((unsigned int)a2 >> 8) * (unsigned int)HIWORD(a2);
LABEL_20:
      if ( (a2 & 6) == 2 )
      {
        v4 = 0;
LABEL_22:
        v11 = (a2 >> 24) & 0xFFFFFF;
        goto LABEL_16;
      }
LABEL_9:
      v7 = a1;
      v8 = BYTE4(a1);
      goto LABEL_10;
    }
LABEL_8:
    v6 = (unsigned int)(HIDWORD(a2) * v5);
    if ( !v3 )
      goto LABEL_9;
    goto LABEL_20;
  }
  v4 = 1;
  if ( v2 )
    goto LABEL_24;
LABEL_13:
  v6 = HIDWORD(a2);
LABEL_14:
  v11 = 0;
  if ( v3 && (a2 & 6) == 2 )
    goto LABEL_22;
LABEL_16:
  v7 = a1;
  v8 = BYTE4(a1);
  if ( (_BYTE)v4 )
  {
    v9 = (v6 << 45) & 0x1FFFE00000000000LL | ((unsigned __int64)v11 << 21);
    return (2 * v4) | 4 | (8 * (v9 | v8 | (32LL * v7)));
  }
LABEL_10:
  v4 = 0;
  v9 = v6 << 29;
  return (2 * v4) | 4 | (8 * (v9 | v8 | (32LL * v7)));
}
