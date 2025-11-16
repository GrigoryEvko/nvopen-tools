// Function: sub_C2E4C0
// Address: 0xc2e4c0
//
__int64 __fastcall sub_C2E4C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // rbx
  __int64 v10; // [rsp+0h] [rbp-C0h]
  _QWORD v11[4]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v12[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v13[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v14[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v15; // [rsp+70h] [rbp-50h]
  __int64 v16; // [rsp+78h] [rbp-48h]
  _QWORD *v17; // [rsp+80h] [rbp-40h]

  switch ( a3 )
  {
    case 0LL:
      goto LABEL_2;
    case 4LL:
      if ( *(_DWORD *)a2 == 1819107705 )
      {
LABEL_2:
        LODWORD(a3) = 1;
LABEL_3:
        v5 = *(_BYTE *)(a1 + 8);
        *(_DWORD *)a1 = a3;
        *(_BYTE *)(a1 + 8) = v5 & 0xFC | 2;
        return a1;
      }
      break;
    case 11LL:
      if ( *(_QWORD *)a2 == 0x7274732D6C6D6179LL && *(_WORD *)(a2 + 8) == 24948 )
      {
        a3 = 2;
        if ( *(_BYTE *)(a2 + 10) == 98 )
          goto LABEL_3;
      }
      break;
    default:
      if ( a3 == 9 && *(_QWORD *)a2 == 0x6165727473746962LL )
      {
        a3 = 3;
        if ( *(_BYTE *)(a2 + 8) == 109 )
          goto LABEL_3;
      }
      break;
  }
  LOBYTE(v13[0]) = 0;
  v10 = sub_2241E50(a1, a2, a3, a4, a5);
  v12[0] = v13;
  v16 = 0x100000000LL;
  v12[1] = 0;
  memset(&v14[1], 0, 24);
  v15 = 0;
  v14[0] = &unk_49DD210;
  v17 = v12;
  sub_CB5980(v14, 0, 0, 0);
  v11[1] = "Unknown remark format: '%s'";
  v11[2] = a2;
  v11[0] = &unk_49DBDF0;
  sub_CB6620(v14, v11);
  v14[0] = &unk_49DD210;
  sub_CB5840(v14);
  v14[0] = v12;
  LOWORD(v15) = 260;
  v6 = sub_22077B0(64);
  v7 = v6;
  if ( v6 )
    sub_C63EB0(v6, v14, 22, v10);
  v8 = v7 & 0xFFFFFFFFFFFFFFFELL;
  if ( (_QWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0], v13[0] + 1LL);
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v8;
  return a1;
}
