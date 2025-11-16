// Function: sub_16825F0
// Address: 0x16825f0
//
bool __fastcall sub_16825F0(__int64 a1, void *a2)
{
  int v3; // edx
  int v4; // eax
  int v5; // edi
  ssize_t v6; // rax
  int v7; // eax
  fd_set readfds; // [rsp+0h] [rbp-A0h] BYREF

  memset(&readfds, 0, sizeof(readfds));
  v3 = *(_DWORD *)(a1 + 188);
  v4 = v3 + 63;
  if ( v3 >= 0 )
    v4 = *(_DWORD *)(a1 + 188);
  readfds.fds_bits[v4 >> 6] |= 1LL << (*(_DWORD *)(a1 + 188) % 64);
  v5 = *(_DWORD *)(a1 + 196);
  readfds.fds_bits[v5 / 64] |= 1LL << (((v5 + ((unsigned int)(*(int *)(a1 + 196) >> 31) >> 26)) & 0x3F)
                                     - ((unsigned int)(*(int *)(a1 + 196) >> 31) >> 26));
  while ( 1 )
  {
    if ( v3 >= v5 )
      v5 = v3;
    if ( select(v5 + 1, &readfds, 0, 0, 0) != 1 )
    {
      v7 = *__errno_location();
LABEL_11:
      *(_DWORD *)(a1 + 4) = v7;
      _InterlockedCompareExchange((volatile signed __int32 *)a1, 11, 0);
      return 0;
    }
    if ( !*(_BYTE *)(a1 + 205) )
    {
      v6 = read(*(_DWORD *)(a1 + 188), a2, 1u);
      if ( v6 != -1 )
        return v6 == 1;
    }
    if ( *(_BYTE *)(a1 + 205) )
      return 0;
    v7 = *__errno_location();
    if ( v7 != 11 )
      goto LABEL_11;
    v3 = *(_DWORD *)(a1 + 188);
    v5 = *(_DWORD *)(a1 + 196);
  }
}
