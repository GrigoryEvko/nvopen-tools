// Function: sub_134EF80
// Address: 0x134ef80
//
unsigned __int64 __fastcall sub_134EF80(_QWORD *a1)
{
  unsigned __int64 v1; // rbx
  char v2; // dl
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_BYTE *)(v1 + 23);
  if ( (*a1 & 4) == 0 )
  {
    if ( v2 < 0 )
    {
      v10 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v10 + v11;
      if ( *(char *)(v1 + 23) >= 0 )
      {
        if ( (unsigned int)(v12 >> 4) )
          goto LABEL_19;
      }
      else if ( (unsigned int)((v12 - sub_1648A40(v1)) >> 4) )
      {
        if ( *(char *)(v1 + 23) >= 0 )
          goto LABEL_19;
        v13 = *(_DWORD *)(sub_1648A40(v1) + 8);
        if ( *(char *)(v1 + 23) >= 0 )
          goto LABEL_18;
        v14 = sub_1648A40(v1);
        return v1 + -72 - 24LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
      }
    }
    return v1 - 72;
  }
  if ( v2 < 0 )
  {
    v3 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    v5 = v3 + v4;
    if ( *(char *)(v1 + 23) >= 0 )
    {
      if ( (unsigned int)(v5 >> 4) )
        goto LABEL_19;
    }
    else if ( (unsigned int)((v5 - sub_1648A40(v1)) >> 4) )
    {
      if ( *(char *)(v1 + 23) < 0 )
      {
        v6 = *(_DWORD *)(sub_1648A40(v1) + 8);
        if ( *(char *)(v1 + 23) < 0 )
        {
          v7 = sub_1648A40(v1);
          return v1 + -24 - 24LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
        }
LABEL_18:
        BUG();
      }
LABEL_19:
      BUG();
    }
  }
  return v1 - 24;
}
