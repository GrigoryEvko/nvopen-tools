// Function: sub_167C240
// Address: 0x167c240
//
__int64 __fastcall sub_167C240(__int64 a1, bool *a2, __int64 a3, __int64 a4)
{
  char v4; // al
  bool v7; // dl
  unsigned int v8; // eax
  unsigned int v9; // r13d
  unsigned __int8 v10; // al
  unsigned int v11; // ecx
  bool v13; // r9
  bool v14; // al
  char v15; // al
  __int64 v16; // r14
  unsigned __int64 v17; // r15
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rdi
  bool v22; // [rsp+Fh] [rbp-C1h]
  _QWORD v23[2]; // [rsp+10h] [rbp-C0h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+30h] [rbp-A0h]
  const char *v26; // [rsp+40h] [rbp-90h]
  char v27; // [rsp+50h] [rbp-80h]
  char v28; // [rsp+51h] [rbp-7Fh]
  _QWORD v29[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v30; // [rsp+70h] [rbp-60h]
  _BYTE v31[80]; // [rsp+80h] [rbp-50h] BYREF

  v4 = *(_BYTE *)(a4 + 32) & 0xF;
  if ( v4 == 6 )
    goto LABEL_16;
  if ( v4 == 1 )
  {
    if ( (*(_BYTE *)(a3 + 32) & 0xF) != 1 )
    {
      LOBYTE(v9) = sub_15E4F60(a3);
      goto LABEL_14;
    }
LABEL_19:
    if ( (*(_BYTE *)(a4 + 33) & 3) != 1 )
      goto LABEL_20;
    LOBYTE(v9) = 1;
LABEL_30:
    *a2 = v9;
    return 0;
  }
  v7 = sub_15E4F60(a4);
  if ( (*(_BYTE *)(a3 + 32) & 0xF) == 1 )
  {
    if ( !v7 )
      goto LABEL_16;
    goto LABEL_19;
  }
  v22 = v7;
  LOBYTE(v8) = sub_15E4F60(a3);
  v9 = v8;
  if ( !v22 )
  {
    if ( !(_BYTE)v8 )
    {
      v10 = *(_BYTE *)(a4 + 32) & 0xF;
      if ( v10 == 10 )
      {
        v15 = *(_BYTE *)(a3 + 32) & 0xF;
        if ( ((v15 + 14) & 0xFu) > 3 )
        {
          if ( v15 == 10 )
          {
            v16 = sub_1632FA0(*(_QWORD *)(a3 + 40));
            v17 = sub_12BE0A0(v16, *(_QWORD *)(a3 + 24));
            *a2 = v17 < sub_12BE0A0(v16, *(_QWORD *)(a4 + 24));
            return v9;
          }
          goto LABEL_11;
        }
      }
      else
      {
        v11 = v10 - 4;
        if ( v11 <= 1 || v10 == 2 )
        {
          if ( (*(_BYTE *)(a3 + 32) & 0xFu) - 2 > 1 || v11 > 1 )
            goto LABEL_11;
        }
        else
        {
          if ( ((v10 + 7) & 0xFu) <= 1 || v10 == 3 )
          {
LABEL_11:
            *a2 = 0;
            return v9;
          }
          v18 = *(_BYTE *)(a3 + 32) & 0xF;
          if ( ((v18 + 14) & 0xFu) > 3 && ((v18 + 7) & 0xFu) > 1 )
          {
            v28 = 1;
            v26 = "': symbol multiply defined!";
            v27 = 3;
            v23[0] = sub_1649960(a4);
            v29[0] = "Linking globals named '";
            v29[1] = v23;
            v23[1] = v19;
            LOWORD(v30) = 1283;
            v24[1] = "': symbol multiply defined!";
            v24[0] = v29;
            LOWORD(v25) = 770;
            v20 = **(_QWORD **)(a1 + 8);
            sub_1670450((__int64)v31, 0, (__int64)v24);
            v21 = v20;
            v9 = 1;
            sub_16027F0(v21, (__int64)v31);
            return v9;
          }
        }
      }
    }
LABEL_16:
    *a2 = 1;
    return 0;
  }
LABEL_14:
  if ( (*(_BYTE *)(a4 + 33) & 3) == 1 )
    goto LABEL_30;
  if ( (*(_BYTE *)(a3 + 32) & 0xF) == 9 )
    goto LABEL_16;
LABEL_20:
  v13 = sub_15E4F60(a4);
  v14 = 0;
  if ( !v13 )
    v14 = sub_15E4F60(a3);
  *a2 = v14;
  return 0;
}
