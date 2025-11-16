// Function: sub_222C080
// Address: 0x222c080
//
__int64 __fastcall sub_222C080(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  bool v5; // zf
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  int v9; // eax
  char *v10; // rsi
  char *v11; // rdx
  __int64 v12; // rdx
  int v13; // eax
  __int64 result; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // ecx
  char v19[25]; // [rsp+1h] [rbp-19h] BYREF

  if ( (*(_BYTE *)(a1 + 120) & 0x11) == 0 )
    return 0xFFFFFFFFLL;
  if ( *(_BYTE *)(a1 + 169) )
  {
    if ( *(_BYTE *)(a1 + 192) )
    {
      v5 = *(_QWORD *)(a1 + 16) == *(_QWORD *)(a1 + 8);
      *(_BYTE *)(a1 + 192) = 0;
      v6 = *(_QWORD *)(a1 + 184);
      v7 = *(_QWORD *)(a1 + 152);
      v8 = *(_QWORD *)(a1 + 176) + !v5;
      *(_QWORD *)(a1 + 176) = v8;
      *(_QWORD *)(a1 + 8) = v7;
      *(_QWORD *)(a1 + 16) = v8;
      *(_QWORD *)(a1 + 24) = v6;
    }
    v9 = sub_222BE20(a1, a1 + 140);
    if ( sub_222BFB0(a1, v9, 1, *(_QWORD *)(a1 + 140)) == -1 )
      return 0xFFFFFFFFLL;
  }
  v10 = *(char **)(a1 + 32);
  v11 = *(char **)(a1 + 40);
  if ( v10 >= v11 )
  {
    v15 = *(_QWORD *)(a1 + 160);
    if ( v15 > 1 )
    {
      v17 = *(_QWORD *)(a1 + 152);
      v18 = *(_DWORD *)(a1 + 120);
      *(_QWORD *)(a1 + 8) = v17;
      *(_QWORD *)(a1 + 16) = v17;
      *(_QWORD *)(a1 + 24) = v17;
      if ( (v18 & 0x10) != 0 || (v18 & 1) != 0 )
      {
        *(_QWORD *)(a1 + 40) = v17;
        *(_QWORD *)(a1 + 32) = v17;
        *(_QWORD *)(a1 + 48) = v17 + v15 - 1;
      }
      else
      {
        *(_QWORD *)(a1 + 40) = 0;
        *(_QWORD *)(a1 + 32) = 0;
        *(_QWORD *)(a1 + 48) = 0;
      }
      *(_BYTE *)(a1 + 170) = 1;
      if ( a2 != -1 )
      {
        *(_BYTE *)(*(_QWORD *)(a1 + 40))++ = a2;
        return a2;
      }
    }
    else
    {
      v19[0] = a2;
      if ( a2 != -1 )
      {
        if ( sub_222BCC0(a1, v19, 1u, a4) )
        {
          *(_BYTE *)(a1 + 170) = 1;
          return a2;
        }
        return 0xFFFFFFFFLL;
      }
      *(_BYTE *)(a1 + 170) = 1;
    }
    return 0;
  }
  if ( a2 != -1 )
  {
    *v11 = a2;
    v10 = *(char **)(a1 + 32);
    v11 = (char *)(*(_QWORD *)(a1 + 40) + 1LL);
    *(_QWORD *)(a1 + 40) = v11;
  }
  if ( !sub_222BCC0(a1, v10, v11 - v10, a4) )
    return 0xFFFFFFFFLL;
  v12 = *(_QWORD *)(a1 + 152);
  v13 = *(_DWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 8) = v12;
  *(_QWORD *)(a1 + 16) = v12;
  *(_QWORD *)(a1 + 24) = v12;
  if ( ((v13 & 0x10) != 0 || (v13 & 1) != 0) && (v16 = *(_QWORD *)(a1 + 160), v16 > 1) )
  {
    *(_QWORD *)(a1 + 40) = v12;
    *(_QWORD *)(a1 + 32) = v12;
    *(_QWORD *)(a1 + 48) = v12 + v16 - 1;
  }
  else
  {
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 48) = 0;
  }
  result = a2;
  if ( a2 == -1 )
    return 0;
  return result;
}
