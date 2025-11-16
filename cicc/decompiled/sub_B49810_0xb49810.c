// Function: sub_B49810
// Address: 0xb49810
//
unsigned __int64 __fastcall sub_B49810(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  char v5; // dl
  __int64 v6; // rax
  unsigned __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12

  if ( *(char *)(a1 + 7) >= 0 )
    goto LABEL_30;
  v2 = sub_BD2BC0(a1);
  v4 = v2 + v3;
  v5 = *(_BYTE *)(a1 + 7);
  if ( v5 < 0 )
  {
    v6 = sub_BD2BC0(a1);
    v5 = *(_BYTE *)(a1 + 7);
    if ( v4 - v6 > 112 )
      goto LABEL_4;
    if ( v5 < 0 )
    {
      v13 = sub_BD2BC0(a1);
      v15 = v13 + v14;
      result = *(char *)(a1 + 7) >= 0 ? 0LL : sub_BD2BC0(a1);
      if ( result != v15 )
      {
        while ( *(_DWORD *)(result + 8) > a2 || *(_DWORD *)(result + 12) <= a2 )
        {
          result += 16LL;
          if ( result == v15 )
            goto LABEL_30;
        }
        return result;
      }
    }
LABEL_30:
    BUG();
  }
  if ( v4 <= 112 )
    goto LABEL_30;
LABEL_4:
  result = 0;
  if ( v5 < 0 )
  {
    v8 = sub_BD2BC0(a1);
    if ( *(char *)(a1 + 7) >= 0 )
    {
      v11 = 0;
    }
    else
    {
      v9 = sub_BD2BC0(a1);
      v11 = v9 + v10;
    }
    if ( v11 == v8 )
      return v11;
    while ( 1 )
    {
      result = v8
             + 16LL
             * (((a2 - *(_DWORD *)(v8 + 8)) << 10)
              / (unsigned int)((unsigned int)((*(_DWORD *)(v11 - 4) - *(_DWORD *)(v8 + 8)) << 10)
                             / ((__int64)(v11 - v8) >> 4)));
      if ( result >= v11 )
        result = v11 - 16;
      v12 = *(_DWORD *)(result + 12);
      if ( *(_DWORD *)(result + 8) <= a2 )
        break;
      if ( a2 >= v12 )
      {
LABEL_10:
        v8 = result + 16;
        if ( v11 == result + 16 )
          return result;
      }
      else
      {
        v11 = result;
        if ( result == v8 )
          return result;
      }
    }
    if ( a2 < v12 )
      return result;
    goto LABEL_10;
  }
  return result;
}
