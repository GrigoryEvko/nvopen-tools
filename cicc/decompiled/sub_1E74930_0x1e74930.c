// Function: sub_1E74930
// Address: 0x1e74930
//
_QWORD *__fastcall sub_1E74930(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  _QWORD *v7; // r13
  const char **v9; // rdx
  const char *v10; // [rsp+0h] [rbp-60h] BYREF
  char v11; // [rsp+10h] [rbp-50h]
  char v12; // [rsp+11h] [rbp-4Fh]
  const char *v13; // [rsp+20h] [rbp-40h] BYREF
  const char *v14; // [rsp+28h] [rbp-38h]
  __int16 v15; // [rsp+30h] [rbp-30h]

  v1 = sub_22077B0(584);
  v2 = v1;
  if ( !v1 )
  {
    v7 = (_QWORD *)sub_22077B0(2272);
    if ( !v7 )
      return v7;
    goto LABEL_5;
  }
  *(_QWORD *)(v1 + 8) = a1;
  *(_QWORD *)(v1 + 48) = v1 + 64;
  *(_QWORD *)(v1 + 56) = 0x1000000000LL;
  *(_QWORD *)(v1 + 16) = 0;
  *(_BYTE *)(v1 + 44) = 0;
  *(_QWORD *)v1 = &unk_49FCA80;
  v10 = "TopQ";
  v13 = "TopQ";
  v14 = ".A";
  *(_QWORD *)(v1 + 24) = 0;
  *(_QWORD *)(v1 + 32) = 0;
  *(_DWORD *)(v1 + 40) = 0;
  v12 = 1;
  v11 = 3;
  *(_QWORD *)(v1 + 136) = 0;
  *(_QWORD *)(v1 + 144) = 0;
  *(_QWORD *)(v1 + 152) = 0;
  v15 = 771;
  *(_DWORD *)(v1 + 160) = 1;
  sub_16E2FC0((__int64 *)(v1 + 168), (__int64)&v13);
  v3 = v11;
  *(_QWORD *)(v2 + 200) = 0;
  *(_QWORD *)(v2 + 208) = 0;
  *(_QWORD *)(v2 + 216) = 0;
  if ( v3 )
  {
    if ( v3 == 1 )
    {
      v13 = ".P";
      v15 = 259;
    }
    else
    {
      v9 = (const char **)v10;
      if ( v12 != 1 )
      {
        v9 = &v10;
        v3 = 2;
      }
      v13 = (const char *)v9;
      v14 = ".P";
      LOBYTE(v15) = v3;
      HIBYTE(v15) = 3;
    }
  }
  else
  {
    v15 = 256;
  }
  *(_DWORD *)(v2 + 224) = 4;
  sub_16E2FC0((__int64 *)(v2 + 232), (__int64)&v13);
  *(_QWORD *)(v2 + 328) = v2 + 344;
  *(_QWORD *)(v2 + 336) = 0x1000000000LL;
  *(_QWORD *)(v2 + 424) = v2 + 440;
  *(_QWORD *)(v2 + 432) = 0x1000000000LL;
  *(_QWORD *)(v2 + 264) = 0;
  *(_QWORD *)(v2 + 272) = 0;
  *(_QWORD *)(v2 + 280) = 0;
  *(_QWORD *)(v2 + 288) = 0;
  sub_1E72570(v2 + 136, (__int64)&v13, v2 + 440, v4, v5, v6);
  *(_QWORD *)(v2 + 504) = v2 + 520;
  *(_QWORD *)(v2 + 512) = 0x800000000LL;
  v7 = (_QWORD *)sub_22077B0(2272);
  if ( v7 )
  {
LABEL_5:
    sub_1F03E40(v7, a1[1], a1[2], 1);
    v7[265] = v2;
    *v7 = &unk_49FC890;
    v7[263] = a1[5];
    v7[264] = a1[6];
    sub_1F024C0(v7 + 266, v7 + 6, v7 + 43);
    v7[277] = 0;
    v7[278] = 0;
    v7[279] = 0;
    v7[280] = 0;
    v7[281] = 0;
    v7[282] = 0;
    v7[283] = 0;
    return v7;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2);
  return 0;
}
