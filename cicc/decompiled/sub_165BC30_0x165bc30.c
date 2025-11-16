// Function: sub_165BC30
// Address: 0x165bc30
//
unsigned __int64 __fastcall sub_165BC30(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  char v2; // dl
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_BYTE *)(v1 + 23);
  if ( (*a1 & 4) == 0 )
  {
    if ( v2 >= 0 )
      return v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
    v6 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v8 = v6 + v7;
    if ( *(char *)(v1 + 23) >= 0 )
    {
      if ( (unsigned int)(v8 >> 4) )
        goto LABEL_21;
    }
    else if ( (unsigned int)((v8 - sub_1648A40(v1)) >> 4) )
    {
      if ( *(char *)(v1 + 23) < 0 )
      {
        sub_1648A40(v1);
        if ( *(char *)(v1 + 23) < 0 )
          goto LABEL_7;
LABEL_20:
        BUG();
      }
      goto LABEL_21;
    }
    v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    return v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  }
  if ( v2 >= 0 )
    return v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  v3 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v5 = v3 + v4;
  if ( *(char *)(v1 + 23) >= 0 )
  {
    if ( !(unsigned int)(v5 >> 4) )
      goto LABEL_17;
LABEL_21:
    BUG();
  }
  if ( !(unsigned int)((v5 - sub_1648A40(v1)) >> 4) )
  {
LABEL_17:
    v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    return v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  }
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_21;
  sub_1648A40(v1);
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_20;
LABEL_7:
  sub_1648A40(v1);
  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  return v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
}
