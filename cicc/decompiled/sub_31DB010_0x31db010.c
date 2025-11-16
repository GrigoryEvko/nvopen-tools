// Function: sub_31DB010
// Address: 0x31db010
//
__int64 __fastcall sub_31DB010(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r14
  int v7; // eax
  __int64 v8; // rdx
  char *v9; // rcx
  _QWORD *v10; // r8
  int v12; // eax
  __int64 *v13; // rdi
  __int64 v14; // r14
  __int64 v15; // r13
  unsigned int v16; // eax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  void *v20; // rax
  const char *v21; // [rsp+8h] [rbp-168h]
  unsigned int v22; // [rsp+18h] [rbp-158h]
  _QWORD *v23; // [rsp+18h] [rbp-158h]
  _QWORD *v24; // [rsp+18h] [rbp-158h]
  _QWORD v25[4]; // [rsp+20h] [rbp-150h] BYREF
  __int16 v26; // [rsp+40h] [rbp-130h]
  __int128 v27; // [rsp+50h] [rbp-120h] BYREF
  __int128 v28; // [rsp+60h] [rbp-110h]
  __int64 v29; // [rsp+70h] [rbp-100h]
  __int128 v30; // [rsp+80h] [rbp-F0h]
  char v31; // [rsp+A0h] [rbp-D0h]
  char v32; // [rsp+A1h] [rbp-CFh]
  _OWORD v33[2]; // [rsp+B0h] [rbp-C0h] BYREF
  char v34; // [rsp+D0h] [rbp-A0h]
  char v35; // [rsp+D1h] [rbp-9Fh]
  __int128 v36; // [rsp+E0h] [rbp-90h]
  __int16 v37; // [rsp+100h] [rbp-70h]
  const char *v38[2]; // [rsp+110h] [rbp-60h] BYREF
  __int128 v39; // [rsp+120h] [rbp-50h]
  char v40; // [rsp+130h] [rbp-40h]
  char v41; // [rsp+131h] [rbp-3Fh]

  v3 = sub_31DB000(a1);
  if ( *(_DWORD *)(v3 + 52) != 14 )
    goto LABEL_24;
  v12 = *(_DWORD *)(v3 + 56);
  if ( v12 != 27 )
  {
    if ( v12 )
      goto LABEL_24;
  }
  v13 = *(__int64 **)(a1 + 232);
  v14 = *(_QWORD *)(v13[7] + 8) + 16LL * a2;
  if ( *(_BYTE *)(v14 + 9) )
    goto LABEL_24;
  v15 = sub_2E79000(v13);
  LOBYTE(v16) = sub_2E7A190(v14, v15);
  v17 = *(_QWORD *)v14;
  v22 = v16;
  LOBYTE(v38[0]) = *(_BYTE *)(v14 + 8);
  v18 = sub_31DA6B0(a1);
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, const char **))(*(_QWORD *)v18 + 64LL))(
          v18,
          v15,
          v22,
          v17,
          v38);
  if ( *(_DWORD *)(v19 + 144) || (v10 = *(_QWORD **)(v19 + 160)) == 0 )
  {
LABEL_24:
    v4 = sub_31DA930(a1);
    LODWORD(v36) = a2;
    v5 = *(_QWORD *)(a1 + 216);
    v37 = 265;
    v6 = v4;
    v32 = 1;
    *(_QWORD *)&v30 = "_";
    v31 = 3;
    v7 = sub_31DA6A0(a1);
    switch ( *(_DWORD *)(v6 + 24) )
    {
      case 0:
        v8 = 0;
        v9 = (char *)byte_3F871B3;
        break;
      case 1:
      case 3:
        v8 = 2;
        v9 = ".L";
        break;
      case 2:
      case 4:
        v8 = 1;
        v9 = "L";
        break;
      case 5:
        v8 = 2;
        v9 = "L#";
        break;
      case 6:
        v8 = 1;
        v9 = "$";
        break;
      case 7:
        v8 = 3;
        v9 = "L..";
        break;
      default:
        BUG();
    }
    LODWORD(v28) = v7;
    v25[0] = v9;
    v25[1] = v8;
    v25[2] = &unk_44D3674;
    v26 = 773;
    *(_QWORD *)&v27 = v25;
    LOWORD(v29) = 2306;
    v35 = v31;
    *(_QWORD *)&v33[0] = &v27;
    v33[1] = v30;
    v34 = 2;
    v38[0] = (const char *)v33;
    v39 = v36;
    v38[1] = v21;
    v40 = 2;
    v41 = v37;
    return sub_E6C460(v5, v38);
  }
  else if ( !*v10 )
  {
    if ( (*((_BYTE *)v10 + 9) & 0x70) != 0x20
      || *((char *)v10 + 8) < 0
      || (*((_BYTE *)v10 + 8) |= 8u, v23 = v10, v20 = sub_E807D0(v10[3]), v10 = v23, (*v23 = v20) == 0) )
    {
      v24 = v10;
      (*(void (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
        *(_QWORD *)(a1 + 224),
        v10,
        9);
      return (__int64)v24;
    }
  }
  return (__int64)v10;
}
