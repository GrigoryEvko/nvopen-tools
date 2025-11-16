// Function: sub_B7EA20
// Address: 0xb7ea20
//
__int64 __fastcall sub_B7EA20(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  unsigned int v4; // r14d
  __int64 v5; // rdi
  __int64 (*v6)(); // rax

  if ( *(_DWORD *)(a1 + 200) )
  {
    v3 = 0;
    v4 = 0;
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v3);
        v6 = *(__int64 (**)())(*(_QWORD *)v5 + 24LL);
        if ( v6 != sub_97DD00 )
          break;
        if ( *(_DWORD *)(a1 + 200) <= ++v3 )
          return v4;
      }
      ++v3;
      v4 |= ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
    }
    while ( *(_DWORD *)(a1 + 200) > v3 );
  }
  else
  {
    return 0;
  }
  return v4;
}
