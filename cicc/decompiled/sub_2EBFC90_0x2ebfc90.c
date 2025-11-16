// Function: sub_2EBFC90
// Address: 0x2ebfc90
//
__int64 __fastcall sub_2EBFC90(_QWORD *a1, unsigned int a2)
{
  _QWORD *v3; // r10
  unsigned __int16 *v4; // rax
  unsigned __int16 v5; // cx
  unsigned __int16 v6; // r11
  unsigned __int16 v7; // ax
  _WORD *v8; // rdi
  __int64 v9; // r8

  v3 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  v4 = (unsigned __int16 *)(v3[6] + 4LL * a2);
  v5 = *v4;
  v6 = v4[1];
  if ( !*v4 )
    return 0;
  while ( 1 )
  {
    v7 = v5;
    v8 = (_WORD *)(v3[7] + 2LL * *(unsigned int *)(v3[1] + 24LL * v5 + 8));
    if ( !v8 )
      return 1;
    v9 = a1[48];
    if ( (*(_QWORD *)(v9 + 8LL * (v5 >> 6)) & (1LL << v5)) != 0 )
    {
      do
      {
        v7 += *v8;
        if ( !*v8 )
          return 1;
        ++v8;
      }
      while ( (*(_QWORD *)(v9 + 8LL * (v7 >> 6)) & (1LL << v7)) != 0 );
    }
    v5 = v6;
    v6 = 0;
    if ( !v5 )
      return 0;
  }
}
