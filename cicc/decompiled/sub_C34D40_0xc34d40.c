// Function: sub_C34D40
// Address: 0xc34d40
//
__int64 __fastcall sub_C34D40(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // cl
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdx

  v2 = *(_BYTE *)(a2 + 20);
  v3 = v2 & 7;
  if ( (v2 & 7) == 1 )
  {
    LOWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 30;
    v6 = *(_QWORD *)sub_C33930(a2) & 0x3FFLL;
    v2 = *(_BYTE *)(a2 + 20);
  }
  else if ( v3 == 3 )
  {
    v6 = 0;
    LOWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
  }
  else if ( v3 )
  {
    LODWORD(v4) = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 14;
    v5 = *(_QWORD *)sub_C33930(a2);
    if ( (int)v4 == 1 )
      v4 = (*(_QWORD *)sub_C33930(a2) >> 10) & 1LL;
    v2 = *(_BYTE *)(a2 + 20);
    v6 = v5 & 0x3FF;
  }
  else
  {
    LOWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 30;
    v6 = 0;
  }
  *(_DWORD *)(a1 + 8) = 16;
  *(_QWORD *)a1 = ((_WORD)v4 << 10) & 0x7C00 | v6 | ((unsigned __int64)((v2 & 8) != 0) << 15);
  return a1;
}
