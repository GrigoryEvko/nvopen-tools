// Function: sub_1D013C0
// Address: 0x1d013c0
//
__int16 __fastcall sub_1D013C0(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  void (*v7)(void); // rdx
  bool v8; // cf

  v3 = *(_DWORD **)(a1 + 704);
  LODWORD(v4) = v3[2];
  if ( !(_DWORD)v4 )
    return v4;
  v4 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
    return v4;
  v4 = *(unsigned __int16 *)(v4 + 24);
  if ( (_WORD)v4 == 193 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)v3 + 32LL);
    if ( (void (*)())v4 == nullsub_678 )
      return v4;
    goto LABEL_16;
  }
  if ( (__int16)v4 > 193 )
  {
    if ( (_WORD)v4 == 194 )
      return v4;
    v8 = (_WORD)v4 == 239;
    LOWORD(v4) = v4 - 239;
    if ( v8 || (_WORD)v4 == 1 )
      return v4;
  }
  else if ( (unsigned __int16)(v4 - 2) <= 0x31u )
  {
    v5 = 0x8C00000000004LL;
    if ( _bittest64(&v5, v4) )
      return v4;
  }
  v6 = *(_QWORD *)v3;
  if ( (*(_BYTE *)(a2 + 228) & 2) != 0 )
  {
    v7 = *(void (**)(void))(v6 + 32);
    if ( v7 != nullsub_678 )
    {
      v7();
      v6 = **(_QWORD **)(a1 + 704);
    }
  }
  v4 = *(_QWORD *)(v6 + 40);
  if ( (void (*)())v4 != nullsub_679 )
LABEL_16:
    LOWORD(v4) = ((__int64 (*)(void))v4)();
  return v4;
}
