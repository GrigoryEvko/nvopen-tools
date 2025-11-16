// Function: sub_165A000
// Address: 0x165a000
//
unsigned __int64 __fastcall sub_165A000(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  char v2; // dl
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rdx

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_BYTE *)(v1 + 23);
  v3 = v1;
  if ( (*a1 & 4) == 0 )
  {
    if ( v2 >= 0 )
    {
      v10 = -72;
      return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
    }
    v11 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v13 = v11 + v12;
    if ( *(char *)(v1 + 23) >= 0 )
    {
      if ( (unsigned int)(v13 >> 4) )
        goto LABEL_22;
    }
    else if ( (unsigned int)((v13 - sub_1648A40(v1)) >> 4) )
    {
      if ( *(char *)(v1 + 23) < 0 )
      {
        v14 = *(_DWORD *)(sub_1648A40(v1) + 8);
        if ( *(char *)(v1 + 23) < 0 )
        {
          v15 = sub_1648A40(v1);
          v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
          return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
        }
LABEL_21:
        BUG();
      }
      goto LABEL_22;
    }
    v10 = -72;
    v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
  }
  if ( v2 >= 0 )
  {
    v10 = -24;
    return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
  }
  v4 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = v4 + v5;
  if ( *(char *)(v1 + 23) >= 0 )
  {
    if ( !(unsigned int)(v6 >> 4) )
      goto LABEL_18;
LABEL_22:
    BUG();
  }
  if ( !(unsigned int)((v6 - sub_1648A40(v1)) >> 4) )
  {
LABEL_18:
    v10 = -24;
    v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
  }
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_22;
  v7 = *(_DWORD *)(sub_1648A40(v1) + 8);
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_21;
  v8 = sub_1648A40(v1);
  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
  return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v3 + v10 + 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) - v1) >> 3);
}
