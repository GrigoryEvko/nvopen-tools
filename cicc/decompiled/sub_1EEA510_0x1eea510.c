// Function: sub_1EEA510
// Address: 0x1eea510
//
unsigned __int64 __fastcall sub_1EEA510(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 i; // rdx
  unsigned __int64 result; // rax
  __int64 v5; // rdx

  v1 = *(_QWORD *)(a1 + 32);
  sub_2103FD0(a1 + 96, v1);
  v2 = *(_QWORD *)(a1 + 48);
  for ( i = v2 + 16LL * *(unsigned int *)(a1 + 56); v2 != i; *(_QWORD *)(v2 - 8) = 0 )
  {
    while ( *(_QWORD *)(v2 + 8) != v1 )
    {
      v2 += 16;
      if ( v2 == i )
        goto LABEL_6;
    }
    *(_DWORD *)(v2 + 4) = 0;
    v2 += 16;
  }
LABEL_6:
  result = *(_QWORD *)(a1 + 32);
  if ( result == *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) )
  {
    *(_QWORD *)(a1 + 32) = 0;
    *(_BYTE *)(a1 + 44) = 0;
  }
  else
  {
    result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
    if ( !result )
      BUG();
    v5 = *(_QWORD *)result;
    if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v5 = *(_QWORD *)result;
      }
    }
    *(_QWORD *)(a1 + 32) = result;
  }
  return result;
}
