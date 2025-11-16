// Function: sub_160CB70
// Address: 0x160cb70
//
__int64 __fastcall sub_160CB70(__int64 a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v4; // r14d
  __int64 v5; // r12
  __int64 v6; // rdi

  v2 = *(_DWORD *)(a1 + 192) - 1;
  if ( v2 < 0 )
  {
    return 0;
  }
  else
  {
    v4 = 0;
    v5 = 8LL * v2;
    do
    {
      --v2;
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + v5);
      v5 -= 8;
      v4 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 32LL))(v6, a2);
    }
    while ( v2 != -1 );
  }
  return v4;
}
