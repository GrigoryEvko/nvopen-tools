// Function: sub_AC2D70
// Address: 0xac2d70
//
__int64 __fastcall sub_AC2D70(_BYTE *a1)
{
  unsigned int v1; // r13d
  char v2; // dl
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // r12
  signed int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rdx
  _BYTE *v11; // r13
  _BYTE *v12; // rdi
  _BYTE *v13; // r12
  _BYTE *v14; // rax

  v1 = 2;
  v2 = *a1;
  if ( *a1 <= 3u )
    return v1;
  v4 = a1;
  while ( v2 == 4 )
  {
    v4 = (_QWORD *)*(v4 - 8);
    v2 = *(_BYTE *)v4;
    if ( *(_BYTE *)v4 <= 3u )
      return 2;
  }
  v5 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
  if ( v2 == 5 && *((_WORD *)v4 + 1) == 15 )
  {
    v9 = v4[-4 * v5];
    v10 = v4[4 * (1 - v5)];
    if ( *(_BYTE *)v9 == 5 && *(_BYTE *)v10 == 5 && *(_WORD *)(v9 + 2) == 47 && *(_WORD *)(v10 + 2) == 47 )
    {
      v11 = *(_BYTE **)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
      v12 = *(_BYTE **)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
      if ( *v11 == 4 && *v12 == 4 && *((_QWORD *)v12 - 8) == *((_QWORD *)v11 - 8) )
        return 0;
      v13 = (_BYTE *)sub_BD4070(v12);
      if ( *v13 <= 3u )
      {
        v14 = (_BYTE *)sub_BD4070(v11);
        if ( *v14 > 3u )
        {
          if ( *v14 == 6 && (v13[33] & 0x40) != 0 )
            return 1;
        }
        else if ( (v14[33] & 0x40) != 0 )
        {
          v1 = 1;
          if ( (v13[33] & 0x40) != 0 )
            return v1;
        }
      }
      v5 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
    }
  }
  v6 = 32 * v5;
  if ( (*((_BYTE *)v4 + 7) & 0x40) != 0 )
  {
    v7 = (_QWORD *)*(v4 - 1);
    v4 = &v7[(unsigned __int64)v6 / 8];
  }
  else
  {
    v7 = &v4[v6 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v1 = 0;
  if ( v7 == v4 )
    return v1;
  do
  {
    v8 = sub_AC2D70(*v7);
    if ( v8 >= (int)v1 )
      v1 = v8;
    v7 += 4;
  }
  while ( v4 != v7 );
  return v1;
}
