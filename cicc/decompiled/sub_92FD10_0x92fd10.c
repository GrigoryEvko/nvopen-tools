// Function: sub_92FD10
// Address: 0x92fd10
//
_DWORD *__fastcall sub_92FD10(__int64 a1, unsigned int *a2)
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
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 528) + 32LL);
      if ( (*(_BYTE *)(v4 + 193) & 0x10) != 0 )
        v3 = *(_DWORD *)(v4 + 64);
    }
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 368LL);
    sub_941230(v5, v3);
    return (_DWORD *)sub_93FCC0(v5, a1 + 48);
  }
  return result;
}
