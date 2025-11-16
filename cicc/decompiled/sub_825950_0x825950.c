// Function: sub_825950
// Address: 0x825950
//
_BOOL8 __fastcall sub_825950(__int64 a1, _BYTE *a2, int a3, _DWORD *a4)
{
  char v4; // al
  char v7; // si
  char v9; // r8
  char v10; // cl
  char v11; // di
  char v12; // r8
  char v13; // r9
  _BOOL4 v14; // r13d
  _BOOL4 v16; // esi
  char *v17; // r12
  char *v18; // rax
  char *v19; // r12
  char *v20; // rax
  char *v21; // r12
  char *v22; // rax
  char *v23; // r12
  char *v24; // rax

  v4 = *(_BYTE *)(a1 + 197);
  if ( (v4 & 4) != 0 || (*(_BYTE *)(a1 - 8) & 0x10) != 0 || !a2 )
    return 1;
  v7 = a2[198];
  v9 = *(_BYTE *)(a1 + 198);
  v10 = v9 & 0x18;
  v11 = v7 & 0x18;
  v12 = v9 & 0x30;
  v13 = v7 & 0x30;
  if ( v12 == 16 )
  {
    if ( v10 != 16 )
    {
      if ( v13 != 16 && v11 != 16 )
      {
        if ( (v4 & 0x18) == 0 && !sub_825770((__int64)a2) && (a2[199] & 2) == 0 )
        {
          if ( a3 )
          {
            v23 = sub_8258E0(a1, 0);
            v24 = sub_8258E0((__int64)a2, 1);
            sub_6865F0((a2[193] & 2) == 0 ? 3468 : 3470, dword_4F07508, (__int64)v24, (__int64)v23);
          }
          else
          {
            sub_684B30(((a2[193] & 2) != 0) + 3471, dword_4F07508);
          }
        }
        if ( a4 )
          *a4 = 1;
      }
      return 1;
    }
LABEL_13:
    v16 = (v7 & 0x20) != 0;
    if ( v12 == 16 )
    {
      if ( !v16 && v13 != 16 && v11 != 16 )
      {
        if ( a3 )
        {
          if ( (v4 & 0x10) == 0 )
          {
            v14 = sub_825770((__int64)a2);
            if ( !v14 && (a2[199] & 2) == 0 )
            {
              v21 = sub_8258E0(a1, 0);
              v22 = sub_8258E0((__int64)a2, 1);
              sub_686610(3474 - ((a2[193] & 2) == 0), dword_4F07508, (__int64)v22, (__int64)v21);
              return v14;
            }
          }
        }
        return 0;
      }
    }
    else if ( !v16 && v13 != 16 && v11 != 16 )
    {
      if ( a3 )
      {
        if ( (v4 & 0x10) == 0 )
        {
          v14 = sub_825770((__int64)a2);
          if ( !v14 && (a2[199] & 2) == 0 )
          {
            v17 = sub_8258E0(a1, 0);
            v18 = sub_8258E0((__int64)a2, 1);
            sub_686610(3476 - ((a2[193] & 2) == 0), dword_4F07508, (__int64)v18, (__int64)v17);
            return v14;
          }
        }
      }
      return 0;
    }
    return 1;
  }
  if ( v10 == 16 )
    goto LABEL_13;
  if ( v13 == 16 && v11 == 16 )
  {
    if ( a3 )
    {
      if ( (v4 & 0x10) == 0 )
      {
        v14 = sub_825770((__int64)a2);
        if ( !v14 && (a2[199] & 2) == 0 )
        {
          v19 = sub_8258E0(a1, 0);
          v20 = sub_8258E0((__int64)a2, 1);
          sub_686610(((a2[193] & 2) != 0) + 3464, dword_4F07508, (__int64)v20, (__int64)v19);
          return v14;
        }
      }
    }
    return 0;
  }
  return 1;
}
