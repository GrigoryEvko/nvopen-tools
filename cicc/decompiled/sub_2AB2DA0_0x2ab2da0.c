// Function: sub_2AB2DA0
// Address: 0x2ab2da0
//
__int64 __fastcall sub_2AB2DA0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v4; // rcx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  int v8; // esi
  int v9; // ebx
  unsigned int i; // esi
  __int64 v11; // r10
  unsigned int v12; // esi

  v4 = (unsigned int)a3;
  v5 = HIDWORD(a3);
  if ( BYTE4(a3) )
  {
    v6 = *(_QWORD *)(a1 + 200);
    v7 = *(unsigned int *)(a1 + 216);
    if ( (_DWORD)v7 )
    {
      v8 = 37 * v4 - 1;
LABEL_5:
      v9 = 1;
      for ( i = (v7 - 1) & v8; ; i = (v7 - 1) & v12 )
      {
        v11 = v6 + 72LL * i;
        if ( *(_DWORD *)v11 == (_DWORD)v4 && (_BYTE)v5 == *(_BYTE *)(v11 + 4) )
        {
          v6 += 72LL * i;
          return sub_B19060(v6 + 8, a2, v6, v4);
        }
        if ( *(_DWORD *)v11 == -1 && *(_BYTE *)(v11 + 4) )
          break;
        v12 = v9 + i;
        ++v9;
      }
      v6 += 72 * v7;
    }
    return sub_B19060(v6 + 8, a2, v6, v4);
  }
  if ( (_DWORD)a3 != 1 )
  {
    v6 = *(_QWORD *)(a1 + 200);
    v7 = *(unsigned int *)(a1 + 216);
    if ( (_DWORD)v7 )
    {
      v8 = 37 * v4;
      goto LABEL_5;
    }
    return sub_B19060(v6 + 8, a2, v6, v4);
  }
  return 1;
}
