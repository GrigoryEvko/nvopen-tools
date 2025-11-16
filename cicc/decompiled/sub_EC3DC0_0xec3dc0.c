// Function: sub_EC3DC0
// Address: 0xec3dc0
//
__int64 __fastcall sub_EC3DC0(__int64 a1, char *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdi
  char v9; // al
  _QWORD v10[4]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v11; // [rsp+20h] [rbp-50h]
  _QWORD v12[4]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v13; // [rsp+50h] [rbp-20h]

  v2 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  v3 = v2;
  if ( *(_DWORD *)v2 == 2 )
  {
    v5 = *(_QWORD *)(v2 + 8);
    v4 = *(_QWORD *)(v2 + 16);
    if ( v4 != 8 )
      goto LABEL_6;
  }
  else
  {
    v4 = *(_QWORD *)(v2 + 16);
    v5 = *(_QWORD *)(v3 + 8);
    if ( !v4 )
      goto LABEL_10;
    v6 = v4 - 1;
    if ( !v6 )
      v6 = 1;
    ++v5;
    v4 = v6 - 1;
    if ( v4 != 8 )
    {
LABEL_6:
      switch ( v4 )
      {
        case 7LL:
          if ( *(_DWORD *)v5 == 1668508004 && *(_WORD *)(v5 + 4) == 29281 && *(_BYTE *)(v5 + 6) == 100 )
          {
            v9 = 2;
            goto LABEL_14;
          }
          if ( *(_DWORD *)v5 == 1735549292 && *(_WORD *)(v5 + 4) == 29541 && *(_BYTE *)(v5 + 6) == 116 )
          {
            *a2 = 6;
            goto LABEL_15;
          }
          break;
        case 9LL:
          if ( *(_QWORD *)v5 == 0x7A69735F656D6173LL && *(_BYTE *)(v5 + 8) == 101 )
          {
            v9 = 3;
            goto LABEL_14;
          }
          break;
        case 13LL:
          if ( *(_QWORD *)v5 == 0x6E6F635F656D6173LL && *(_DWORD *)(v5 + 8) == 1953391988 && *(_BYTE *)(v5 + 12) == 115 )
          {
            v9 = 4;
            goto LABEL_14;
          }
          break;
        case 11LL:
          if ( *(_QWORD *)v5 == 0x746169636F737361LL && *(_WORD *)(v5 + 8) == 30313 && *(_BYTE *)(v5 + 10) == 101 )
          {
            v9 = 5;
            goto LABEL_14;
          }
          break;
        default:
          if ( v4 == 6 && *(_DWORD *)v5 == 1702323566 && *(_WORD *)(v5 + 4) == 29811 )
          {
            *a2 = 7;
            goto LABEL_15;
          }
          break;
      }
LABEL_10:
      v10[2] = v5;
      *a2 = 0;
      v7 = *(_QWORD *)(a1 + 8);
      v10[3] = v4;
      v11 = 1283;
      v10[0] = "unrecognized COMDAT type '";
      v12[0] = v10;
      v13 = 770;
      v12[2] = "'";
      return sub_ECE0E0(v7, v12, 0, 0);
    }
  }
  if ( *(_QWORD *)v5 != 0x796C6E6F5F656E6FLL )
    goto LABEL_10;
  v9 = 1;
LABEL_14:
  *a2 = v9;
LABEL_15:
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  return 0;
}
