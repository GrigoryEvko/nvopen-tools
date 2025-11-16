// Function: sub_89A690
// Address: 0x89a690
//
__int64 __fastcall sub_89A690(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r8
  unsigned int v4; // eax
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v9; // rax
  unsigned int v10; // [rsp+0h] [rbp-50h]
  unsigned int v11; // [rsp+4h] [rbp-4Ch]
  int v12; // [rsp+8h] [rbp-48h]
  __int16 v13; // [rsp+Ch] [rbp-44h]
  unsigned int v14; // [rsp+10h] [rbp-40h]
  unsigned __int16 v15; // [rsp+18h] [rbp-38h]
  __int64 *v16; // [rsp+18h] [rbp-38h]

  v3 = a1;
  if ( dword_4F04C44 != -1
    || (v9 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v9 + 6) & 6) != 0)
    || *(_BYTE *)(v9 + 4) == 12
    || (v16 = a3, v4 = sub_89A370(a3), a3 = v16, v3 = a1, v4) )
  {
    v4 = 2052;
    if ( (*(_BYTE *)(a2 + 56) & 2) == 0 )
      return *(_QWORD *)(a2 + 80);
  }
  else if ( (*(_BYTE *)(a2 + 56) & 2) == 0 )
  {
    return *(_QWORD *)(a2 + 80);
  }
  sub_865840(*(_QWORD *)(a2 + 128), 0, 0, 0, v3, (__int64)a3, v4);
  v10 = dword_4F063F8;
  v11 = word_4F063FC[0];
  v12 = dword_4F07508[0];
  v13 = dword_4F07508[1];
  v14 = dword_4F061D8;
  v15 = word_4F061DC[0];
  sub_7BC160(a2 + 96);
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 64) + 104LL);
  v6 = sub_679730(v5, 0, &dword_4F063F8);
  dword_4F07508[0] = v12;
  LOWORD(dword_4F07508[1]) = v13;
  word_4F061DC[0] = v15;
  dword_4F063F8 = v10;
  word_4F063FC[0] = v11;
  dword_4F061D8 = v14;
  sub_863FE0(v5, 0, v7, v14, v11, (__int64 *)v10);
  return v6;
}
