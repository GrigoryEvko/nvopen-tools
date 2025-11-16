// Function: sub_169B370
// Address: 0x169b370
//
unsigned __int64 __fastcall sub_169B370(__int64 a1, char a2)
{
  _WORD *v2; // rax
  void *v3; // r12
  int v4; // eax
  __int64 v5; // r14
  int v6; // ebx
  unsigned __int64 result; // rax

  v2 = *(_WORD **)a1;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF0 | (8 * a2 + 2) & 0xF;
  *(_WORD *)(a1 + 16) = *v2;
  v3 = (void *)sub_1698470(a1);
  v4 = sub_1698310(a1);
  v5 = (unsigned int)(v4 - 1);
  v6 = v4 << 6;
  memset(v3, 255, 8 * v5);
  result = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 - *(_BYTE *)(*(_QWORD *)a1 + 4LL));
  if ( (unsigned int)(v6 - *(_DWORD *)(*(_QWORD *)a1 + 4LL)) >= 0x40 )
    result = 0;
  *((_QWORD *)v3 + v5) = result;
  return result;
}
