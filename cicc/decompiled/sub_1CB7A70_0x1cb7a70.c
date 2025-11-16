// Function: sub_1CB7A70
// Address: 0x1cb7a70
//
__int64 __fastcall sub_1CB7A70(unsigned int *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  int v4; // eax
  unsigned __int64 v5; // r14
  int v6; // eax
  int v7; // edx
  int v8; // ebx
  int v9; // eax

  if ( *(_BYTE *)(a2 + 16) != 78 )
    return 0;
  v3 = *(_QWORD *)(a2 - 24);
  if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
  {
    v4 = *(_DWORD *)(v3 + 36);
    if ( (v4 == 4046 || v4 == 4242)
      && (v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15) )
    {
      v8 = sub_1CB76C0(a1, v5);
      if ( v8 != (unsigned int)sub_1CB76C0(a1, a2) )
      {
        v9 = sub_1CB76C0(a1, v5);
        sub_1CB7560(a1, a2, v9);
        return 1;
      }
    }
    else
    {
      v6 = sub_1CB76C0(a1, a2);
      v7 = a1[1];
      if ( v6 != v7 )
      {
        sub_1CB7560(a1, a2, v7);
        return 1;
      }
    }
  }
  return 0;
}
