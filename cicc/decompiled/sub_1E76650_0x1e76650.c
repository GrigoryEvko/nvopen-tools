// Function: sub_1E76650
// Address: 0x1e76650
//
__int64 *__fastcall sub_1E76650(_QWORD *a1)
{
  __int64 v1; // rax
  const char *v2; // rbx
  char v3; // al
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  char v7; // al
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 *v12; // r12
  const char *v13; // rdi
  char *v14; // rsi
  const char *v16; // rdx
  const char *v17; // rdx
  _QWORD v18[2]; // [rsp+0h] [rbp-70h] BYREF
  char v19; // [rsp+10h] [rbp-60h]
  char v20; // [rsp+11h] [rbp-5Fh]
  const char *v21; // [rsp+20h] [rbp-50h] BYREF
  const char *v22; // [rsp+28h] [rbp-48h]
  __int16 v23; // [rsp+30h] [rbp-40h]

  v1 = sub_22077B0(976);
  v2 = (const char *)v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = a1;
    *(_QWORD *)(v1 + 48) = v1 + 64;
    *(_QWORD *)(v1 + 56) = 0x1000000000LL;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)v1 = &unk_49FC9E0;
    v18[0] = "TopQ";
    v21 = "TopQ";
    *(_QWORD *)(v1 + 32) = 0;
    *(_DWORD *)(v1 + 40) = 0;
    *(_BYTE *)(v1 + 44) = 0;
    *(_QWORD *)(v1 + 128) = 0;
    *(_DWORD *)(v1 + 136) = 0;
    *(_BYTE *)(v1 + 140) = 0;
    v20 = 1;
    v19 = 3;
    *(_QWORD *)(v1 + 144) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    v22 = ".A";
    v23 = 771;
    *(_DWORD *)(v1 + 168) = 1;
    sub_16E2FC0((__int64 *)(v1 + 176), (__int64)&v21);
    v3 = v19;
    *((_QWORD *)v2 + 26) = 0;
    *((_QWORD *)v2 + 27) = 0;
    *((_QWORD *)v2 + 28) = 0;
    if ( v3 )
    {
      if ( v3 == 1 )
      {
        v21 = ".P";
        v23 = 259;
      }
      else
      {
        v17 = (const char *)v18[0];
        if ( v20 != 1 )
        {
          v17 = (const char *)v18;
          v3 = 2;
        }
        v21 = v17;
        v22 = ".P";
        LOBYTE(v23) = v3;
        HIBYTE(v23) = 3;
      }
    }
    else
    {
      v23 = 256;
    }
    *((_DWORD *)v2 + 58) = 4;
    sub_16E2FC0((__int64 *)v2 + 30, (__int64)&v21);
    *((_QWORD *)v2 + 42) = v2 + 352;
    *((_QWORD *)v2 + 43) = 0x1000000000LL;
    *((_QWORD *)v2 + 54) = v2 + 448;
    *((_QWORD *)v2 + 55) = 0x1000000000LL;
    *((_QWORD *)v2 + 34) = 0;
    *((_QWORD *)v2 + 35) = 0;
    *((_QWORD *)v2 + 36) = 0;
    *((_QWORD *)v2 + 37) = 0;
    sub_1E72570((__int64)(v2 + 144), (__int64)&v21, (__int64)(v2 + 448), v4, v5, v6);
    v20 = 1;
    v23 = 771;
    v18[0] = "BotQ";
    v21 = "BotQ";
    v19 = 3;
    *((_QWORD *)v2 + 64) = 0;
    *((_QWORD *)v2 + 65) = 0;
    *((_QWORD *)v2 + 66) = 0;
    v22 = ".A";
    *((_DWORD *)v2 + 134) = 2;
    sub_16E2FC0((__int64 *)v2 + 68, (__int64)&v21);
    v7 = v19;
    *((_QWORD *)v2 + 72) = 0;
    *((_QWORD *)v2 + 73) = 0;
    *((_QWORD *)v2 + 74) = 0;
    if ( v7 )
    {
      if ( v7 == 1 )
      {
        v21 = ".P";
        v23 = 259;
      }
      else
      {
        v16 = (const char *)v18[0];
        if ( v20 != 1 )
        {
          v16 = (const char *)v18;
          v7 = 2;
        }
        v21 = v16;
        v22 = ".P";
        LOBYTE(v23) = v7;
        HIBYTE(v23) = 3;
      }
    }
    else
    {
      v23 = 256;
    }
    *((_DWORD *)v2 + 150) = 8;
    sub_16E2FC0((__int64 *)v2 + 76, (__int64)&v21);
    *((_QWORD *)v2 + 88) = v2 + 720;
    *((_QWORD *)v2 + 89) = 0x1000000000LL;
    *((_QWORD *)v2 + 100) = v2 + 816;
    *((_QWORD *)v2 + 101) = 0x1000000000LL;
    *((_QWORD *)v2 + 80) = 0;
    *((_QWORD *)v2 + 81) = 0;
    *((_QWORD *)v2 + 82) = 0;
    *((_QWORD *)v2 + 83) = 0;
    sub_1E72570((__int64)(v2 + 512), (__int64)&v21, (__int64)(v2 + 816), v8, v9, v10);
    *((_BYTE *)v2 + 880) = 0;
    *(_QWORD *)(v2 + 884) = 0;
    *((_QWORD *)v2 + 112) = 0;
    *((_QWORD *)v2 + 113) = 0;
    *((_DWORD *)v2 + 228) = 0;
    *((_WORD *)v2 + 458) = 0;
    *((_QWORD *)v2 + 115) = 0;
    *((_BYTE *)v2 + 928) = 0;
    *(_QWORD *)(v2 + 932) = 0;
    *((_QWORD *)v2 + 121) = 0;
    *((_QWORD *)v2 + 118) = 0;
    *((_QWORD *)v2 + 119) = 0;
    *((_DWORD *)v2 + 240) = 0;
    *((_WORD *)v2 + 482) = 0;
  }
  v21 = v2;
  v11 = sub_22077B0(4072);
  v12 = (__int64 *)v11;
  if ( v11 )
    sub_1E6E680(v11, a1, (__int64 *)&v21);
  if ( v21 )
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v21 + 16LL))(v21);
  sub_1E722C0(&v21);
  v13 = v21;
  if ( v21 )
  {
    v14 = (char *)v12[278];
    if ( v14 == (char *)v12[279] )
    {
      sub_1E764A0(v12 + 277, v14, &v21);
      v13 = v21;
      if ( !v21 )
        return v12;
    }
    else
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = v21;
        v12[278] += 8;
        return v12;
      }
      v12[278] = 8;
    }
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v13 + 16LL))(v13);
  }
  return v12;
}
