// Function: sub_6790F0
// Address: 0x6790f0
//
__int64 __fastcall sub_6790F0(__int64 a1, int a2, int a3, int a4, int a5)
{
  unsigned int v6; // r14d
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int16 v13; // ax
  __int64 v16; // [rsp+18h] [rbp-1A8h] BYREF
  _BYTE v17[352]; // [rsp+20h] [rbp-1A0h] BYREF
  int v18; // [rsp+180h] [rbp-40h]
  __int16 v19; // [rsp+184h] [rbp-3Ch]

  memset(v17, 0, sizeof(v17));
  v17[67] = 1;
  v6 = dword_4F06650[0];
  v18 = 0;
  v19 = 0;
  if ( a2 )
  {
    v17[28] = 1;
    v17[75] = 1;
    sub_7ADF70(a1, 1);
    ++v17[44];
    v16 = *(_QWORD *)&dword_4F063F8;
    sub_7BDB60(1);
    v12 = unk_4D041C4 == 0 ? 3 : 1;
    if ( a5 )
      v12 = (unsigned int)v12 | 4;
    sub_7C64E0(0, v17, v12);
    sub_7BDC00();
    v13 = word_4F06418[0];
    if ( (unsigned __int16)(word_4F06418[0] - 74) <= 1u )
    {
      sub_6851C0(2379, &v16);
      v13 = word_4F06418[0];
    }
    sub_7AE700(unk_4F061C0 + 24LL, v6, dword_4F06650[0], v13 == 9, a1);
    sub_7AE340(a1);
  }
  else
  {
    if ( dword_4F04C64 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0 )
      v17[76] = 1;
    v17[28] = 1;
    v17[75] = 1;
    sub_7ADF70(a1, 1);
    v16 = *(_QWORD *)&dword_4F063F8;
    sub_7BDB60(1);
    if ( a5 )
      sub_7C64E0(0, v17, 5);
    else
      sub_7C64E0(0, v17, 1);
    sub_7BDC00();
    v8 = dword_4F06650[0];
    sub_7AE700(unk_4F061C0 + 24LL, v6, dword_4F06650[0], word_4F06418[0] == 9, a1);
    sub_7AE340(a1);
    v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v9 + 6) & 2) != 0 && !a4 || a3 && unk_4F04C48 == -1 )
    {
      v10 = *(_QWORD *)(v9 + 624);
      if ( !v10 || !*(_QWORD *)(v10 + 368) || (*(_BYTE *)(v10 + 133) & 8) != 0 )
      {
        v11 = sub_888280(0, 0, v6, v8 - (v6 != v8));
        *(_BYTE *)(v11 + 49) = 1;
        *(_BYTE *)(v11 + 50) = *(_QWORD *)(a1 + 8) == 0;
      }
    }
  }
  return sub_7AE210(a1);
}
