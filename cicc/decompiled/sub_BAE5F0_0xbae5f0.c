// Function: sub_BAE5F0
// Address: 0xbae5f0
//
__int64 __fastcall sub_BAE5F0(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r8
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  int v8; // eax
  _BYTE *v9; // rdi
  unsigned int v10; // ecx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  char v13; // r10
  __int64 v14; // r9

  v4 = a1 + 16;
  v6 = a2;
  if ( a2 <= 9 )
  {
    *(_QWORD *)a1 = v4;
    sub_2240A50(a1, 1, 0, a4, v4);
    v9 = *(_BYTE **)a1;
LABEL_14:
    *v9 = v6 + 48;
    return a1;
  }
  if ( a2 <= 0x63 )
  {
    *(_QWORD *)a1 = v4;
    sub_2240A50(a1, 2, 0, a4, v4);
    v9 = *(_BYTE **)a1;
  }
  else
  {
    if ( a2 <= 0x3E7 )
    {
      a2 = 3;
    }
    else if ( a2 <= 0x270F )
    {
      a2 = 4;
    }
    else
    {
      v7 = a2;
      LODWORD(a2) = 1;
      while ( 1 )
      {
        a4 = v7;
        v8 = a2;
        a2 = (unsigned int)(a2 + 4);
        v7 /= 0x2710u;
        if ( a4 <= 0x1869F )
          break;
        if ( a4 <= 0xF423F )
        {
          *(_QWORD *)a1 = v4;
          a2 = (unsigned int)(v8 + 5);
          goto LABEL_11;
        }
        if ( a4 <= (unsigned __int64)&loc_98967F )
        {
          a2 = (unsigned int)(v8 + 6);
          break;
        }
        if ( a4 <= 0x5F5E0FF )
        {
          a2 = (unsigned int)(v8 + 7);
          break;
        }
      }
    }
    *(_QWORD *)a1 = v4;
LABEL_11:
    sub_2240A50(a1, a2, 0, a4, v4);
    v9 = *(_BYTE **)a1;
    v10 = *(_DWORD *)(a1 + 8) - 1;
    do
    {
      v11 = v6
          - 20 * (v6 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v6 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v12 = v6;
      v6 /= 0x64u;
      v13 = a00010203040506_0[2 * v11 + 1];
      LOBYTE(v11) = a00010203040506_0[2 * v11];
      v9[v10] = v13;
      v14 = v10 - 1;
      v10 -= 2;
      v9[v14] = v11;
    }
    while ( v12 > 0x270F );
    if ( v12 <= 0x3E7 )
      goto LABEL_14;
  }
  v9[1] = a00010203040506_0[2 * v6 + 1];
  *v9 = a00010203040506_0[2 * v6];
  return a1;
}
