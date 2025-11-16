// Function: sub_89A4B0
// Address: 0x89a4b0
//
__int64 __fastcall sub_89A4B0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r8
  int v5; // edx
  char v6; // al
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v10; // rax
  int v11; // eax
  unsigned int v12; // [rsp+0h] [rbp-50h]
  unsigned int v13; // [rsp+4h] [rbp-4Ch]
  int v14; // [rsp+8h] [rbp-48h]
  __int16 v15; // [rsp+Ch] [rbp-44h]
  unsigned int v16; // [rsp+10h] [rbp-40h]
  unsigned int v17; // [rsp+18h] [rbp-38h]
  __int64 *v18; // [rsp+18h] [rbp-38h]

  v4 = a1;
  v5 = 1;
  if ( dword_4F04C44 == -1 )
  {
    v10 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v10 + 6) & 6) == 0 && *(_BYTE *)(v10 + 4) != 12 )
    {
      v18 = a3;
      v11 = sub_89A370(a3);
      v4 = a1;
      a3 = v18;
      LOBYTE(v5) = v11 != 0;
    }
  }
  v6 = *(_BYTE *)(a2 + 56);
  if ( (v6 & 2) == 0 )
    return *(_QWORD *)(a2 + 80);
  if ( dword_4F60174 != unk_4D042F0 )
  {
    ++dword_4F60174;
    if ( (v6 & 4) != 0 )
    {
      *(_BYTE *)(a2 + 56) = v6 & 0xFB;
      v7 = 2050;
    }
    else
    {
      v7 = (v5 << 31 >> 31) & 0x804;
    }
    sub_865840(*(_QWORD *)(a2 + 128), 0, 0, 0, v4, (__int64)a3, v7);
    v12 = dword_4F063F8;
    v13 = word_4F063FC[0];
    v14 = dword_4F07508[0];
    v15 = dword_4F07508[1];
    v16 = dword_4F061D8;
    v17 = word_4F061DC[0];
    sub_7BC160(a2 + 96);
    v8 = sub_6796C0();
    word_4F061DC[0] = v17;
    dword_4F07508[0] = v14;
    LOWORD(dword_4F07508[1]) = v15;
    dword_4F063F8 = v12;
    word_4F063FC[0] = v13;
    dword_4F061D8 = v16;
    sub_863FE0(a2 + 96, 0, v17, v16, v13, (__int64 *)v12);
    --dword_4F60174;
    return v8;
  }
  sub_6851C0(0x3FCu, dword_4F07508);
  return sub_72C930();
}
