// Function: sub_31DF6E0
// Address: 0x31df6e0
//
unsigned __int64 __fastcall sub_31DF6E0(__int64 a1)
{
  unsigned int v2; // [rsp+Ah] [rbp-16h]
  unsigned __int16 v3; // [rsp+Eh] [rbp-12h]

  LOWORD(v2) = sub_31DF670(a1);
  BYTE2(v2) = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 8LL);
  HIBYTE(v2) = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 224) + 8LL) + 1906LL);
  LOBYTE(v3) = *(_BYTE *)(a1 + 976);
  return ((unsigned __int64)v3 << 32) | v2;
}
