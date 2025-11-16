// Function: sub_160D420
// Address: 0x160d420
//
__int64 __fastcall sub_160D420(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  unsigned int v4; // r14d
  __int64 v5; // rdi
  __int64 (*v6)(); // rax

  if ( *(_DWORD *)(a1 + 32) )
  {
    v3 = 0;
    v4 = 0;
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * v3);
        v6 = *(__int64 (**)())(*(_QWORD *)v5 + 24LL);
        if ( v6 != sub_134C070 )
          break;
        if ( *(_DWORD *)(a1 + 32) <= ++v3 )
          return v4;
      }
      ++v3;
      v4 |= ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
    }
    while ( *(_DWORD *)(a1 + 32) > v3 );
  }
  else
  {
    return 0;
  }
  return v4;
}
