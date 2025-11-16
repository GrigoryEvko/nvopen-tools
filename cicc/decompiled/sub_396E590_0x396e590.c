// Function: sub_396E590
// Address: 0x396e590
//
__int64 __fastcall sub_396E590(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  char *v8; // rdx
  __int64 v9; // r13
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned int v17; // [rsp+4h] [rbp-12Ch]
  __int64 v18; // [rsp+8h] [rbp-128h]
  _QWORD v19[2]; // [rsp+10h] [rbp-120h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-110h] BYREF
  __int16 v21; // [rsp+30h] [rbp-100h]
  __int64 v22; // [rsp+40h] [rbp-F0h]
  _QWORD v23[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+70h] [rbp-C0h]
  char *v25; // [rsp+80h] [rbp-B0h]
  char v26; // [rsp+90h] [rbp-A0h]
  char v27; // [rsp+91h] [rbp-9Fh]
  _QWORD v28[2]; // [rsp+A0h] [rbp-90h] BYREF
  char v29; // [rsp+B0h] [rbp-80h]
  char v30; // [rsp+B1h] [rbp-7Fh]
  unsigned __int64 v31; // [rsp+C0h] [rbp-70h]
  __int16 v32; // [rsp+D0h] [rbp-60h]
  unsigned __int128 v33; // [rsp+E0h] [rbp-50h] BYREF
  char v34; // [rsp+F0h] [rbp-40h]
  char v35; // [rsp+F1h] [rbp-3Fh]

  v3 = sub_396E580(a1);
  if ( *(_DWORD *)(v3 + 52) != 15 )
    goto LABEL_20;
  if ( *(_DWORD *)(v3 + 56) != 14 )
    goto LABEL_20;
  v11 = *(_QWORD *)(a1 + 264);
  v12 = *(_QWORD *)(*(_QWORD *)(v11 + 64) + 8LL) + 16LL * a2;
  if ( *(int *)(v12 + 8) < 0 )
    goto LABEL_20;
  v13 = sub_1E0A0C0(v11);
  v17 = sub_1E0AB90(v12, v13);
  v18 = *(_QWORD *)v12;
  LODWORD(v33) = *(_DWORD *)(v12 + 8);
  v14 = sub_396DD80(a1);
  v15 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, unsigned __int128 *))(*(_QWORD *)v14 + 40LL))(
          v14,
          v13,
          v17,
          v18,
          &v33);
  if ( !*(_DWORD *)(v15 + 144) && (v9 = *(_QWORD *)(v15 + 176)) != 0 )
  {
    if ( (*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8
        || (*(_BYTE *)(v9 + 8) |= 4u,
            v16 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24)),
            *(_QWORD *)v9 = v16 | *(_QWORD *)v9 & 7LL,
            !v16) )
      {
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
          *(_QWORD *)(a1 + 256),
          v9,
          8);
      }
    }
  }
  else
  {
LABEL_20:
    v4 = sub_396DDB0(a1);
    LODWORD(v31) = a2;
    v5 = *(_QWORD *)(a1 + 248);
    v32 = 265;
    v6 = v4;
    v27 = 1;
    v25 = "_";
    v26 = 3;
    LODWORD(v22) = sub_396DD70(a1);
    switch ( *(_DWORD *)(v6 + 16) )
    {
      case 0:
        v7 = 0;
        v8 = (char *)byte_3F871B3;
        break;
      case 1:
      case 3:
        v7 = 2;
        v8 = ".L";
        break;
      case 2:
      case 4:
        v7 = 1;
        v8 = "L";
        break;
      case 5:
        v7 = 1;
        v8 = "$";
        break;
    }
    v19[1] = v7;
    v20[0] = v19;
    v20[1] = &unk_44D3674;
    v23[0] = v20;
    v19[0] = v8;
    v23[1] = v22;
    v21 = 773;
    LOWORD(v24) = 2306;
    v28[1] = v25;
    v28[0] = v23;
    v29 = 2;
    v30 = v26;
    v33 = __PAIR128__(v31, v28);
    v34 = 2;
    v35 = v32;
    return sub_38BF510(v5, (__int64)&v33);
  }
  return v9;
}
