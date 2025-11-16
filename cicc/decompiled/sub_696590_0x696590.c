// Function: sub_696590
// Address: 0x696590
//
__int64 __fastcall sub_696590(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax
  char v5; // al
  int v6; // r14d
  int v7; // r13d
  int v8; // eax
  int v9; // [rsp+Ch] [rbp-64h] BYREF
  __int128 v10; // [rsp+10h] [rbp-60h] BYREF
  _OWORD v11[5]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(v2 + 8);
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D3A70(v3) )
    {
      v9 = 0;
      return sub_840360((int)v2 + 8, a2, 0, 0, 0, 0, 0, 0, 0, (__int64)&v10, (__int64)&v9, 0);
    }
    else
    {
      v5 = *(_BYTE *)(v2 + 24);
      v6 = 0;
      v7 = v2 + 152;
      v10 = 0;
      v11[0] = 0;
      if ( v5 != 2 )
        v7 = 0;
      LOBYTE(v6) = v5 == 2;
      v11[1] = 0;
      v8 = sub_6EB660(v2 + 8);
      return sub_8E1010(
               *(_QWORD *)(v2 + 8),
               v6,
               (*(_BYTE *)(v2 + 27) & 0x10) != 0,
               v8,
               0,
               v7,
               a2,
               0,
               0,
               0,
               0,
               (__int64)v11 + 8,
               0);
    }
  }
  else
  {
    result = 1;
    if ( a2 != v3 )
      return (unsigned int)sub_8DED30(v3, a2, 3) != 0;
  }
  return result;
}
