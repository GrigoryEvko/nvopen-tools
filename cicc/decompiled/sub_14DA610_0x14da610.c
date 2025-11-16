// Function: sub_14DA610
// Address: 0x14da610
//
__int64 __fastcall sub_14DA610(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  char v2; // dl
  int v3; // ebx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edx

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_BYTE *)(v1 + 23);
  v3 = *(_DWORD *)(v1 + 20) & 0xFFFFFFF;
  if ( (*a1 & 4) == 0 )
  {
    if ( v2 < 0 )
    {
      v11 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
      v13 = v11 + v12;
      if ( *(char *)(v1 + 23) >= 0 )
      {
        if ( (unsigned int)(v13 >> 4) )
          goto LABEL_21;
      }
      else if ( (unsigned int)((v13 - sub_1648A40(v1)) >> 4) )
      {
        if ( *(char *)(v1 + 23) >= 0 )
          goto LABEL_21;
        v14 = *(_DWORD *)(sub_1648A40(v1) + 8);
        if ( *(char *)(v1 + 23) >= 0 )
LABEL_20:
          BUG();
        v15 = sub_1648A40(v1);
        v17 = *(_DWORD *)(v15 + v16 - 4) - v14;
        return (unsigned int)(v3 - 3 - v17);
      }
    }
    v17 = 0;
    return (unsigned int)(v3 - 3 - v17);
  }
  if ( v2 >= 0 )
  {
LABEL_18:
    v10 = 0;
    return (unsigned int)(v3 - 1 - v10);
  }
  v4 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = v4 + v5;
  if ( *(char *)(v1 + 23) >= 0 )
  {
    if ( !(unsigned int)(v6 >> 4) )
      goto LABEL_18;
LABEL_21:
    BUG();
  }
  if ( !(unsigned int)((v6 - sub_1648A40(v1)) >> 4) )
    goto LABEL_18;
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_21;
  v7 = *(_DWORD *)(sub_1648A40(v1) + 8);
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_20;
  v8 = sub_1648A40(v1);
  v10 = *(_DWORD *)(v8 + v9 - 4) - v7;
  return (unsigned int)(v3 - 1 - v10);
}
