// Function: sub_39071C0
// Address: 0x39071c0
//
__int64 __fastcall sub_39071C0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  unsigned int v13; // r15d
  const char *v14; // rax
  __int64 v15; // rdi
  __int64 v17; // rsi
  char v18; // cl
  __int64 v19; // rdi
  char v20; // al
  __int64 v21; // rdi
  __int64 v22; // rax
  bool v23; // di
  unsigned int v24; // [rsp+Ch] [rbp-74h]
  _QWORD v25[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h]
  _QWORD v28[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v29; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v25[0] = 0;
  v25[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v25) )
  {
    v19 = *(_QWORD *)(a1 + 8);
    v28[0] = "expected identifier in directive";
    v29 = 259;
    return (unsigned int)sub_3909CF0(v19, v28, 0, 0, v3, v4);
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v28[0] = v25;
  v29 = 261;
  v6 = sub_38BF510(v5, (__int64)v28);
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 37
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 36
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    if ( !*(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113) )
    {
      HIBYTE(v29) = 1;
      v14 = "expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '%<type>' or \"<type>\"";
      goto LABEL_15;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 45 )
    {
      HIBYTE(v29) = 1;
      v14 = "expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '@<type>', '%<type>' or \"<type>\"";
      goto LABEL_15;
    }
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v10 = sub_3909290(v9);
  v11 = *(_QWORD *)(a1 + 8);
  v26 = 0;
  v27 = 0;
  v12 = v10;
  v13 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v11 + 144LL))(v11, &v26);
  if ( !(_BYTE)v13 )
  {
    switch ( v27 )
    {
      case 8LL:
        if ( *(_QWORD *)v26 != 0x434E55465F545453LL && *(_QWORD *)v26 != 0x6E6F6974636E7566LL )
          goto LABEL_42;
        v24 = 1;
        goto LABEL_28;
      case 10LL:
        if ( *(_QWORD *)v26 == 0x454A424F5F545453LL && *(_WORD *)(v26 + 8) == 21571 )
          goto LABEL_60;
        if ( *(_QWORD *)v26 == 0x656A626F5F736C74LL && *(_WORD *)(v26 + 8) == 29795 )
        {
LABEL_71:
          v24 = 4;
          goto LABEL_28;
        }
        if ( *(_QWORD *)v26 == 0x4D4D4F435F545453LL && *(_WORD *)(v26 + 8) == 20047 )
        {
          v24 = 5;
          goto LABEL_28;
        }
        v24 = 3;
        if ( *(_QWORD *)v26 == 0x59544F4E5F545453LL && *(_WORD *)(v26 + 8) == 17744 )
        {
          v24 = 6;
          v18 = 0;
          v17 = 1;
        }
        else
        {
          v17 = 0;
          v18 = 1;
        }
        break;
      case 6LL:
        if ( *(_DWORD *)v26 == 1701470831 && *(_WORD *)(v26 + 4) == 29795 )
        {
LABEL_60:
          v24 = 3;
LABEL_28:
          v18 = 0;
          v17 = 1;
          break;
        }
        if ( *(_DWORD *)v26 != 1835888483 || *(_WORD *)(v26 + 4) != 28271 )
        {
          if ( *(_DWORD *)v26 == 2037673838 && *(_WORD *)(v26 + 4) == 25968 )
          {
            v24 = 6;
            v18 = 0;
            v17 = 1;
          }
          else
          {
            v24 = 1;
            v17 = 0;
            v18 = 1;
          }
LABEL_31:
          v20 = v18 & (v27 == 17);
          goto LABEL_32;
        }
        v24 = 5;
        v18 = 0;
        v17 = 1;
        break;
      case 7LL:
        if ( *(_DWORD *)v26 != 1599362131 || *(_WORD *)(v26 + 4) != 19540 || *(_BYTE *)(v26 + 6) != 83 )
        {
          v24 = 1;
          v23 = 0;
          v17 = 0;
          v18 = 1;
LABEL_43:
          if ( v23 )
          {
            if ( *(_QWORD *)v26 == 0x5F554E475F545453LL
              && *(_DWORD *)(v26 + 8) == 1314211401
              && *(_BYTE *)(v26 + 12) == 67 )
            {
LABEL_57:
              v24 = 2;
              goto LABEL_38;
            }
            v20 = v27 == 17;
LABEL_32:
            if ( v20 )
            {
              if ( *(_QWORD *)v26 ^ 0x71696E755F756E67LL | *(_QWORD *)(v26 + 8) ^ 0x63656A626F5F6575LL
                || *(_BYTE *)(v26 + 16) != 116 )
              {
                goto LABEL_34;
              }
              v24 = 7;
              goto LABEL_38;
            }
LABEL_33:
            if ( !(_BYTE)v17 )
            {
LABEL_34:
              v21 = *(_QWORD *)(a1 + 8);
              v28[0] = "unsupported attribute in '.type' directive";
              v29 = 259;
              return (unsigned int)sub_3909790(v21, v12, v28, 0, 0);
            }
LABEL_38:
            if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 8) + 40LL))(
                                 *(_QWORD *)(a1 + 8),
                                 v17,
                                 v26)
                             + 8) == 9 )
            {
              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
              v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
              (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v22 + 256LL))(v22, v6, v24);
              return v13;
            }
            HIBYTE(v29) = 1;
            v14 = "unexpected token in '.type' directive";
            goto LABEL_15;
          }
          break;
        }
        goto LABEL_71;
      default:
LABEL_42:
        v24 = 1;
        v18 = 1;
        v23 = v27 == 13;
        v17 = 0;
        goto LABEL_43;
    }
    if ( v27 == 21 && v18 )
    {
      if ( !(*(_QWORD *)v26 ^ 0x69646E695F756E67LL | *(_QWORD *)(v26 + 8) ^ 0x6E75665F74636572LL)
        && *(_DWORD *)(v26 + 16) == 1869182051
        && *(_BYTE *)(v26 + 20) == 110 )
      {
        goto LABEL_57;
      }
      goto LABEL_33;
    }
    goto LABEL_31;
  }
  HIBYTE(v29) = 1;
  v14 = "expected symbol type in directive";
LABEL_15:
  v15 = *(_QWORD *)(a1 + 8);
  v28[0] = v14;
  LOBYTE(v29) = 3;
  return (unsigned int)sub_3909CF0(v15, v28, 0, 0, v7, v8);
}
