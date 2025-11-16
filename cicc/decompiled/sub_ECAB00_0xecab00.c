// Function: sub_ECAB00
// Address: 0xecab00
//
__int64 __fastcall sub_ECAB00(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  unsigned int v9; // r12d
  const char *v10; // rax
  __int64 v11; // rdi
  unsigned int v13; // r15d
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rax
  const char *v17; // [rsp+0h] [rbp-80h] BYREF
  const char *v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 v20; // [rsp+18h] [rbp-68h]
  const char *v21[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v17 = 0;
  v18 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v17) )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v21[0] = "expected identifier";
    v22 = 259;
    return (unsigned int)sub_ECE0E0(v14, v21, 0, 0);
  }
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v22 = 261;
  v21[0] = v17;
  v21[1] = v18;
  v4 = sub_E6C460(v3, v21);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 38
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 37
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    if ( !*(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113) )
    {
      HIBYTE(v22) = 1;
      v10 = "expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '%<type>' or \"<type>\"";
      goto LABEL_15;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 46 )
    {
      HIBYTE(v22) = 1;
      v10 = "expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '@<type>', '%<type>' or \"<type>\"";
      goto LABEL_15;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v6 = sub_ECD690(v5);
  v7 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  v20 = 0;
  v8 = v6;
  v9 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v7 + 192LL))(v7, &v19);
  if ( !(_BYTE)v9 )
  {
    switch ( v20 )
    {
      case 8LL:
        if ( *(_QWORD *)v19 == 0x434E55465F545453LL || *(_QWORD *)v19 == 0x6E6F6974636E7566LL )
        {
          v13 = 2;
LABEL_21:
          if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 8) + 40LL))(
                               *(_QWORD *)(a1 + 8),
                               &v19,
                               v19)
                           + 8) == 9 )
          {
            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
            v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
            (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v16 + 296LL))(v16, v4, v13);
            return v9;
          }
          HIBYTE(v22) = 1;
          v10 = "expected end of directive";
          goto LABEL_15;
        }
        goto LABEL_50;
      case 10LL:
        if ( *(_QWORD *)v19 == 0x454A424F5F545453LL && *(_WORD *)(v19 + 8) == 21571 )
        {
LABEL_26:
          v13 = 4;
          goto LABEL_21;
        }
        if ( *(_QWORD *)v19 == 0x656A626F5F736C74LL && *(_WORD *)(v19 + 8) == 29795 )
        {
LABEL_62:
          v13 = 5;
          goto LABEL_21;
        }
        if ( *(_QWORD *)v19 == 0x4D4D4F435F545453LL && *(_WORD *)(v19 + 8) == 20047 )
          goto LABEL_64;
        if ( *(_QWORD *)v19 != 0x59544F4E5F545453LL || *(_WORD *)(v19 + 8) != 17744 )
          goto LABEL_34;
        break;
      case 6LL:
        if ( *(_DWORD *)v19 == 1701470831 && *(_WORD *)(v19 + 4) == 29795 )
          goto LABEL_26;
        if ( *(_DWORD *)v19 == 1835888483 && *(_WORD *)(v19 + 4) == 28271 )
        {
LABEL_64:
          v13 = 6;
          goto LABEL_21;
        }
        if ( *(_DWORD *)v19 != 2037673838 || *(_WORD *)(v19 + 4) != 25968 )
          goto LABEL_34;
        break;
      case 7LL:
        if ( *(_DWORD *)v19 != 1599362131 || *(_WORD *)(v19 + 4) != 19540 || *(_BYTE *)(v19 + 6) != 83 )
          goto LABEL_34;
        goto LABEL_62;
      case 13LL:
        if ( *(_QWORD *)v19 == 0x5F554E475F545453LL && *(_DWORD *)(v19 + 8) == 1314211401 && *(_BYTE *)(v19 + 12) == 67 )
          goto LABEL_49;
        goto LABEL_34;
      default:
LABEL_50:
        if ( v20 == 21 )
        {
          if ( !(*(_QWORD *)v19 ^ 0x69646E695F756E67LL | *(_QWORD *)(v19 + 8) ^ 0x6E75665F74636572LL)
            && *(_DWORD *)(v19 + 16) == 1869182051
            && *(_BYTE *)(v19 + 20) == 110 )
          {
LABEL_49:
            v13 = 3;
            goto LABEL_21;
          }
        }
        else if ( v20 == 17
               && !(*(_QWORD *)v19 ^ 0x71696E755F756E67LL | *(_QWORD *)(v19 + 8) ^ 0x63656A626F5F6575LL)
               && *(_BYTE *)(v19 + 16) == 116 )
        {
          v13 = 8;
          goto LABEL_21;
        }
LABEL_34:
        v15 = *(_QWORD *)(a1 + 8);
        v21[0] = "unsupported attribute";
        v22 = 259;
        return (unsigned int)sub_ECDA70(v15, v8, v21, 0, 0);
    }
    v13 = 7;
    goto LABEL_21;
  }
  HIBYTE(v22) = 1;
  v10 = "expected symbol type";
LABEL_15:
  v11 = *(_QWORD *)(a1 + 8);
  v21[0] = v10;
  LOBYTE(v22) = 3;
  return (unsigned int)sub_ECE0E0(v11, v21, 0, 0);
}
