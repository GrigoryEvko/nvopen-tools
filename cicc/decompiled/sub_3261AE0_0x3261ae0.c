// Function: sub_3261AE0
// Address: 0x3261ae0
//
__int64 __fastcall sub_3261AE0(unsigned int *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // r13
  unsigned __int64 v4; // r14
  unsigned int v5; // ebx
  unsigned __int64 v6; // rax
  bool v7; // al
  unsigned int v9; // ebx

  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v3 = (_BYTE *)*((_QWORD *)a1 + 1);
  v4 = *a1;
  v5 = *(_DWORD *)(v2 + 32);
  if ( v5 > 0x40 )
  {
    v9 = v5 - sub_C444A0(v2 + 24);
    v7 = 1;
    if ( v9 > 0x40 )
      goto LABEL_4;
    v6 = **(_QWORD **)(v2 + 24);
  }
  else
  {
    v6 = *(_QWORD *)(v2 + 24);
  }
  v7 = v4 <= v6;
LABEL_4:
  *v3 |= v7;
  return 1;
}
