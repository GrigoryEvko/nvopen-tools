// Function: sub_10069D0
// Address: 0x10069d0
//
__int64 __fastcall sub_10069D0(__int64 **a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  void **v6; // rax
  void **v7; // r13
  char v8; // al
  void **v9; // r13
  int v10; // r14d
  unsigned int v11; // r15d
  _BYTE *v12; // rax
  _BYTE *v13; // r13
  char v14; // al
  _BYTE *v15; // r13
  char v16; // [rsp+Fh] [rbp-31h]

  if ( *(_BYTE *)a2 != 18 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    v5 = (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17;
    if ( (unsigned int)v5 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v6 = (void **)sub_AD7630(a2, 0, v5);
    v7 = v6;
    if ( !v6 || *(_BYTE *)v6 != 18 )
    {
      if ( *(_BYTE *)(v4 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v4 + 32);
        if ( v10 )
        {
          v16 = 0;
          v11 = 0;
          while ( 1 )
          {
            v12 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a2, v11);
            v13 = v12;
            if ( !v12 )
              break;
            v14 = *v12;
            if ( v14 != 13 )
            {
              if ( v14 != 18 )
                return 0;
              if ( *((void **)v13 + 3) == sub_C33340() )
              {
                v15 = (_BYTE *)*((_QWORD *)v13 + 4);
                if ( (v15[20] & 7) != 3 )
                  return 0;
              }
              else
              {
                if ( (v13[44] & 7) != 3 )
                  return 0;
                v15 = v13 + 24;
              }
              if ( (v15[20] & 8) != 0 )
                return 0;
              v16 = 1;
            }
            if ( v10 == ++v11 )
            {
              if ( v16 )
                goto LABEL_7;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v6[3] == sub_C33340() )
    {
      v9 = (void **)v7[4];
      if ( (*((_BYTE *)v9 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v8 = *((_BYTE *)v7 + 44);
      v9 = v7 + 3;
      if ( (v8 & 7) != 3 )
        return 0;
    }
    if ( (*((_BYTE *)v9 + 20) & 8) == 0 )
      goto LABEL_7;
    return 0;
  }
  if ( *(void **)(a2 + 24) != sub_C33340() )
  {
    if ( (*(_BYTE *)(a2 + 44) & 7) == 3 && (*(_BYTE *)(a2 + 44) & 8) == 0 )
      goto LABEL_7;
    return 0;
  }
  v3 = *(_QWORD *)(a2 + 32);
  if ( (*(_BYTE *)(v3 + 20) & 7) != 3 || (*(_BYTE *)(v3 + 20) & 8) != 0 )
    return 0;
LABEL_7:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
