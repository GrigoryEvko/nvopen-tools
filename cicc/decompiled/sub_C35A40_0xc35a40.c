// Function: sub_C35A40
// Address: 0xc35a40
//
__int64 __fastcall sub_C35A40(__int64 a1, char a2)
{
  __int64 v2; // rax
  unsigned int v3; // r13d
  __int64 v4; // rax

  if ( a2 && !*(_BYTE *)(*(_QWORD *)a1 + 25LL) )
    BUG();
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF8 | 2;
  sub_C33EE0(a1);
  v2 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF7 | (8 * (a2 & 1));
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(v2 + 4);
  v3 = *(_DWORD *)(v2 + 8) - 1;
  v4 = sub_C33900(a1);
  return sub_C45DB0(v4, v3);
}
