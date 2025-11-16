// Function: sub_C34B20
// Address: 0xc34b20
//
__int64 __fastcall sub_C34B20(__int64 a1, __int64 a2)
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
    LODWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 254;
    v6 = *(_QWORD *)sub_C33930(a2) & 0x7FFFFFLL;
    v2 = *(_BYTE *)(a2 + 20);
  }
  else if ( v3 == 3 )
  {
    v6 = 0;
    LODWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
  }
  else if ( v3 )
  {
    LODWORD(v4) = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 126;
    v5 = *(_QWORD *)sub_C33930(a2);
    if ( (int)v4 == 1 )
      v4 = (*(_QWORD *)sub_C33930(a2) >> 23) & 1LL;
    v2 = *(_BYTE *)(a2 + 20);
    v6 = v5 & 0x7FFFFF;
  }
  else
  {
    LODWORD(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 254;
    v6 = 0;
  }
  *(_DWORD *)(a1 + 8) = 32;
  *(_QWORD *)a1 = ((_DWORD)v4 << 23) & 0x7F800000 | v6 | ((unsigned __int64)((v2 & 8) != 0) << 31);
  return a1;
}
