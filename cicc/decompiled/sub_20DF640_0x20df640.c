// Function: sub_20DF640
// Address: 0x20df640
//
__int64 __fastcall sub_20DF640(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned int i; // r12d
  __int64 v6; // rdi
  __int64 (*v7)(); // rdx
  int v8; // eax

  v2 = *(_QWORD *)(a2 + 24);
  v3 = *(_QWORD *)(v2 + 32);
  for ( i = *(_DWORD *)(*(_QWORD *)(a1 + 232) + 8LL * *(int *)(v2 + 48)); a2 != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(a1 + 472);
      v7 = *(__int64 (**)())(*(_QWORD *)v6 + 128LL);
      v8 = -1;
      if ( v7 != sub_1F39410 )
        v8 = ((__int64 (__fastcall *)(__int64, __int64))v7)(v6, v3);
      i += v8;
      if ( !v3 )
        BUG();
      if ( (*(_BYTE *)v3 & 4) == 0 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( a2 == v3 )
        return i;
    }
    while ( (*(_BYTE *)(v3 + 46) & 8) != 0 )
      v3 = *(_QWORD *)(v3 + 8);
  }
  return i;
}
