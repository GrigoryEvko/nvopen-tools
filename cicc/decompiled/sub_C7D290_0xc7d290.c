// Function: sub_C7D290
// Address: 0xc7d290
//
__int64 __fastcall sub_C7D290(_DWORD *a1, _DWORD *a2)
{
  int *v2; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned int v7; // edx
  unsigned int v8; // edx
  _QWORD *v9; // rax
  int v10; // eax
  __int64 result; // rax
  unsigned __int64 v12; // rsi
  unsigned int v13; // ecx
  unsigned int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rcx
  _QWORD *v17; // rax
  unsigned __int64 v18; // rcx
  unsigned int v19; // eax
  unsigned int v20; // eax
  unsigned int v21; // edx
  __int64 v22; // rsi

  v2 = a1 + 6;
  v4 = a1[5] & 0x3F;
  v5 = (unsigned int)(v4 + 1);
  v6 = (unsigned int)(v4 + 25);
  *((_BYTE *)a1 + v4 + 24) = 0x80;
  v7 = 64 - v5;
  if ( (unsigned __int64)(64 - v5) <= 7 )
  {
    v17 = (_QWORD *)((char *)a1 + v6);
    if ( v7 >= 8 )
    {
      *v17 = 0;
      *(_QWORD *)((char *)v17 + v7 - 8) = 0;
      v18 = (unsigned __int64)(v17 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      v19 = (v7 + (_DWORD)v17 - v18) & 0xFFFFFFF8;
      if ( v19 >= 8 )
      {
        v20 = v19 & 0xFFFFFFF8;
        v21 = 0;
        do
        {
          v22 = v21;
          v21 += 8;
          *(_QWORD *)(v18 + v22) = 0;
        }
        while ( v21 < v20 );
      }
    }
    else if ( (v7 & 4) != 0 )
    {
      *(_DWORD *)v17 = 0;
      *(_DWORD *)((char *)v17 + v7 - 4) = 0;
    }
    else if ( v7 )
    {
      *(_BYTE *)v17 = 0;
      if ( (v7 & 2) != 0 )
        *(_WORD *)((char *)v17 + v7 - 2) = 0;
    }
    sub_C7C890(a1, v2, 64);
    v6 = 24;
    v8 = 56;
  }
  else
  {
    v8 = 56 - v5;
  }
  v9 = (_QWORD *)((char *)a1 + v6);
  if ( v8 >= 8 )
  {
    *v9 = 0;
    *(_QWORD *)((char *)v9 + v8 - 8) = 0;
    v12 = (unsigned __int64)(v9 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    v13 = (v8 + (_DWORD)v9 - v12) & 0xFFFFFFF8;
    if ( v13 >= 8 )
    {
      v14 = 0;
      v15 = v13 & 0xFFFFFFF8;
      do
      {
        v16 = v14;
        v14 += 8;
        *(_QWORD *)(v12 + v16) = 0;
      }
      while ( v14 < v15 );
    }
  }
  else if ( (v8 & 4) != 0 )
  {
    *(_DWORD *)v9 = 0;
    *(_DWORD *)((char *)v9 + v8 - 4) = 0;
  }
  else if ( v8 )
  {
    *(_BYTE *)v9 = 0;
    if ( (v8 & 2) != 0 )
      *(_WORD *)((char *)v9 + v8 - 2) = 0;
  }
  v10 = 8 * a1[5];
  a1[5] = v10;
  a1[20] = v10;
  a1[21] = a1[4];
  sub_C7C890(a1, v2, 64);
  *a2 = *a1;
  a2[1] = a1[1];
  a2[2] = a1[2];
  result = (unsigned int)a1[3];
  a2[3] = result;
  return result;
}
