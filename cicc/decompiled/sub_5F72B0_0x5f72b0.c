// Function: sub_5F72B0
// Address: 0x5f72b0
//
__int64 __fastcall sub_5F72B0(__int64 a1, unsigned __int64 a2, __int64 *a3, int a4, char a5, _QWORD *a6, _DWORD *a7)
{
  _DWORD *v10; // r8
  unsigned int v11; // ecx
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // [rsp+4h] [rbp-4Ch]
  _QWORD *v18; // [rsp+8h] [rbp-48h]
  _BYTE v19[56]; // [rsp+18h] [rbp-38h] BYREF

  v10 = a7;
  *a7 = 0;
  if ( a4 )
  {
    v18 = a6;
    v16 = sub_6935B0(a2, 0xFFFFFFFFLL, v19);
    a6 = v18;
    v10 = a7;
    v11 = v16;
    if ( (a5 & 1) == 0 && a2 && (*(_BYTE *)(a2 + 172) & 1) != 0 && unk_4D041D8 )
    {
      v17 = v16;
      sub_684B30(2908, v18);
      a6 = v18;
      v11 = v17;
      v10 = a7;
    }
  }
  else
  {
    v11 = dword_4F04C64;
    v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( *(_BYTE *)(v12 + 4) == 1 )
    {
      v15 = *(_QWORD *)(v12 + 624);
      if ( v15 )
      {
        if ( (*(_BYTE *)(v15 + 131) & 8) != 0 )
        {
          if ( *(_BYTE *)(v12 - 772) == 9 )
            v11 = dword_4F04C64 - 3;
          else
            v11 = dword_4F04C64 - 2;
        }
      }
    }
  }
  v13 = unk_4D043C4;
  if ( unk_4D043C4 )
  {
    v13 = 0;
    if ( unk_4D03B90 != -1 )
      v13 = (*(_BYTE *)(unk_4D03B98 + 176LL * unk_4D03B90 + 5) & 8) != 0;
  }
  return sub_5F6FE0(a1, a2, a3, v11, a4, a5, a6, v10, v13);
}
