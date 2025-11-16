// Function: sub_127FDC0
// Address: 0x127fdc0
//
_QWORD *__fastcall sub_127FDC0(_QWORD *a1, unsigned __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v5; // rax

  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v4 = sub_8D4AB0(a2);
  else
    v4 = *(_DWORD *)(a2 + 136);
  v5 = sub_127A030(a1[4] + 8LL, a2, 0);
  return sub_127FC40(a1, v5, a3, v4, 0);
}
