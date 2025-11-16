// Function: sub_9CF0C0
// Address: 0x9cf0c0
//
__int64 __fastcall sub_9CF0C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  char v8; // dl
  char v9; // al
  __int64 v10; // rax
  __int64 v12; // rax
  char v13; // al
  unsigned int v14; // [rsp+Ch] [rbp-84h]
  __int64 v15; // [rsp+18h] [rbp-78h] BYREF
  __int64 v16; // [rsp+20h] [rbp-70h] BYREF
  char v17; // [rsp+28h] [rbp-68h]
  __int64 v18[4]; // [rsp+30h] [rbp-60h] BYREF
  char v19; // [rsp+50h] [rbp-40h]
  char v20; // [rsp+51h] [rbp-3Fh]

  v5 = *(_QWORD *)(a3 + 16);
  v14 = *(_DWORD *)(a3 + 32);
  sub_9CDFE0(v18, a3, 32 * a2, a4);
  v7 = v18[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v7;
    return a1;
  }
  sub_9CEA50((__int64)&v16, a3, 0, v6);
  v8 = v17 & 1;
  v9 = (2 * (v17 & 1)) | v17 & 0xFD;
  v17 = v9;
  if ( v8 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v17 = v9 & 0xFD;
    v12 = v16;
    v16 = 0;
    *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
LABEL_11:
    if ( v16 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
    return a1;
  }
  if ( v16 == 0xE00000002LL )
  {
    v13 = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = 8 * v5 - v14;
    *(_BYTE *)(a1 + 8) = v13;
    return a1;
  }
  v20 = 1;
  v18[0] = (__int64)"Expected value symbol table subblock";
  v19 = 3;
  sub_9C8190(&v15, (__int64)v18);
  v10 = v15;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v17 & 2) != 0 )
    sub_9CEF10(&v16);
  if ( (v17 & 1) != 0 )
    goto LABEL_11;
  return a1;
}
