// Function: sub_C35910
// Address: 0xc35910
//
__int64 __fastcall sub_C35910(__int64 a1, char a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // r13
  int v4; // eax
  __int64 v5; // r14
  int v6; // r12d
  unsigned __int64 v7; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)a1;
  if ( a2 && !*(_BYTE *)(v2 + 25) )
    BUG();
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF0 | 2 | (8 * (a2 & 1));
  *(_DWORD *)(a1 + 16) = *(_DWORD *)v2;
  v3 = (_QWORD *)sub_C33900(a1);
  v4 = sub_C337D0(a1);
  v5 = (unsigned int)(v4 - 1);
  v6 = v4;
  memset(v3, 255, 8 * v5);
  v7 = 0xFFFFFFFFFFFFFFFFLL >> (((_BYTE)v6 << 6) - *(_BYTE *)(*(_QWORD *)a1 + 8LL));
  if ( (unsigned int)((v6 << 6) - *(_DWORD *)(*(_QWORD *)a1 + 8LL)) >= 0x40 )
    v7 = 0;
  v3[v5] = v7;
  result = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 1 && *(_DWORD *)(result + 20) == 1 && *(_DWORD *)(result + 8) > 1u )
    *v3 &= ~1uLL;
  return result;
}
