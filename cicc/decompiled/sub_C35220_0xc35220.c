// Function: sub_C35220
// Address: 0xc35220
//
__int64 __fastcall sub_C35220(__int64 a1, __int64 a2)
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
    LOBYTE(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
    v6 = *(_QWORD *)sub_C33930(a2) & 7LL;
    v2 = *(_BYTE *)(a2 + 20);
  }
  else if ( v3 == 3 )
  {
    LOBYTE(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
    v6 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    v4 = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 7;
    v5 = *(_QWORD *)sub_C33930(a2);
    if ( v4 == 1 )
      LOBYTE(v4) = (*(_QWORD *)sub_C33930(a2) & 8LL) != 0;
    v2 = *(_BYTE *)(a2 + 20);
    v6 = v5 & 7;
  }
  *(_DWORD *)(a1 + 8) = 8;
  *(_QWORD *)a1 = (8 * (_BYTE)v4) & 0x78 | v6 | ((unsigned __int64)((v2 & 8) != 0) << 7);
  return a1;
}
