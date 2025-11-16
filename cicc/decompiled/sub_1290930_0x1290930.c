// Function: sub_1290930
// Address: 0x1290930
//
_DWORD *__fastcall sub_1290930(__int64 a1, unsigned int *a2)
{
  _DWORD *result; // rax
  unsigned int v3; // r8d
  __int64 v4; // rax
  __int64 v5; // r12

  result = &dword_4D04658;
  if ( !dword_4D04658 )
  {
    v3 = *a2;
    if ( !*a2 && !*((_WORD *)a2 + 2) )
    {
      v3 = 0;
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 440) + 32LL);
      if ( (*(_BYTE *)(v4 + 193) & 0x10) != 0 )
        v3 = *(_DWORD *)(v4 + 64);
    }
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 384LL);
    sub_12A0360(v5, v3);
    return (_DWORD *)sub_129F080(v5, a1 + 48);
  }
  return result;
}
