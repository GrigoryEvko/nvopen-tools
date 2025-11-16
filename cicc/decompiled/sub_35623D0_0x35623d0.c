// Function: sub_35623D0
// Address: 0x35623d0
//
__int64 __fastcall sub_35623D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r12
  char v6; // dl
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r13

  if ( a2 - a1 <= 0 )
    return a3;
  v3 = a2 - 56;
  v4 = a3 - 56;
  v5 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3);
  do
  {
    sub_C7D6A0(*(_QWORD *)(v4 - 24), 8LL * *(unsigned int *)(v4 - 8), 8);
    *(_DWORD *)(v4 - 8) = 0;
    *(_QWORD *)(v4 - 24) = 0;
    *(_DWORD *)(v4 - 16) = 0;
    *(_DWORD *)(v4 - 12) = 0;
    ++*(_QWORD *)(v4 - 32);
    v7 = *(_QWORD *)(v3 - 24);
    ++*(_QWORD *)(v3 - 32);
    v8 = *(_QWORD *)(v4 - 24);
    *(_QWORD *)(v4 - 24) = v7;
    LODWORD(v7) = *(_DWORD *)(v3 - 16);
    *(_QWORD *)(v3 - 24) = v8;
    LODWORD(v8) = *(_DWORD *)(v4 - 16);
    *(_DWORD *)(v4 - 16) = v7;
    LODWORD(v7) = *(_DWORD *)(v3 - 12);
    *(_DWORD *)(v3 - 16) = v8;
    LODWORD(v8) = *(_DWORD *)(v4 - 12);
    *(_DWORD *)(v4 - 12) = v7;
    LODWORD(v7) = *(_DWORD *)(v3 - 8);
    *(_DWORD *)(v3 - 12) = v8;
    LODWORD(v8) = *(_DWORD *)(v4 - 8);
    *(_DWORD *)(v4 - 8) = v7;
    *(_DWORD *)(v3 - 8) = v8;
    if ( v4 != v3 )
    {
      if ( *(_DWORD *)(v3 + 8) )
      {
        if ( *(_QWORD *)v4 != v4 + 16 )
          _libc_free(*(_QWORD *)v4);
        *(_QWORD *)v4 = *(_QWORD *)v3;
        *(_DWORD *)(v4 + 8) = *(_DWORD *)(v3 + 8);
        *(_DWORD *)(v4 + 12) = *(_DWORD *)(v3 + 12);
        *(_QWORD *)v3 = v3 + 16;
        *(_DWORD *)(v3 + 12) = 0;
        *(_DWORD *)(v3 + 8) = 0;
      }
      else
      {
        *(_DWORD *)(v4 + 8) = 0;
      }
    }
    v6 = *(_BYTE *)(v3 + 16);
    v4 -= 88;
    v3 -= 88;
    *(_BYTE *)(v4 + 104) = v6;
    *(_DWORD *)(v4 + 108) = *(_DWORD *)(v3 + 108);
    *(_DWORD *)(v4 + 112) = *(_DWORD *)(v3 + 112);
    *(_DWORD *)(v4 + 116) = *(_DWORD *)(v3 + 116);
    *(_DWORD *)(v4 + 120) = *(_DWORD *)(v3 + 120);
    *(_QWORD *)(v4 + 128) = *(_QWORD *)(v3 + 128);
    *(_DWORD *)(v4 + 136) = *(_DWORD *)(v3 + 136);
    --v5;
  }
  while ( v5 );
  v9 = -8 * ((a2 - a1) >> 3);
  if ( a2 - a1 <= 0 )
    v9 = -88;
  return v9 + a3;
}
