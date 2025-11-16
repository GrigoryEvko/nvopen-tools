// Function: sub_117CBA0
// Address: 0x117cba0
//
bool __fastcall sub_117CBA0(const __m128i *a1, __int64 *a2, __int64 a3, _BYTE *a4, __int64 a5, char a6)
{
  __int64 v10; // rdx
  __int64 *v12; // rax
  __int64 *v13; // rsi
  char v14; // al
  void **v15; // rax
  __int64 v16; // r8
  void **v17; // rcx
  __int64 v18; // rsi
  void **v19; // rax
  void **v20; // rcx
  char v21; // al
  void **v22; // rcx
  char v23; // [rsp+7h] [rbp-49h]
  void **v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  void **v26; // [rsp+10h] [rbp-40h]
  int v27; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)a3 == 18 )
  {
    if ( *(void **)(a3 + 24) == sub_C33340() )
    {
      v10 = *(_QWORD *)(a3 + 32);
      if ( (*(_BYTE *)(v10 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v10 = a3 + 24;
      if ( (*(_BYTE *)(a3 + 44) & 7) != 3 )
        return 0;
    }
    if ( (*(_BYTE *)(v10 + 20) & 8) != 0 )
      return 0;
  }
  else
  {
    v25 = *(_QWORD *)(a3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 > 1 || *(_BYTE *)a3 > 0x15u )
      return 0;
    v15 = (void **)sub_AD7630(a3, 0, a3);
    v16 = v25;
    if ( !v15 || (v26 = v15, *(_BYTE *)v15 != 18) )
    {
      if ( *(_BYTE *)(v16 + 8) == 17 )
      {
        v27 = *(_DWORD *)(v16 + 32);
        if ( v27 )
        {
          v23 = 0;
          v18 = 0;
          while ( 1 )
          {
            v19 = (void **)sub_AD69F0((unsigned __int8 *)a3, v18);
            v20 = v19;
            if ( !v19 )
              break;
            v21 = *(_BYTE *)v19;
            v24 = v20;
            if ( v21 != 13 )
            {
              if ( v21 != 18 )
                return 0;
              if ( v20[3] == sub_C33340() )
              {
                v22 = (void **)v24[4];
                if ( (*((_BYTE *)v22 + 20) & 7) != 3 )
                  return 0;
              }
              else
              {
                if ( (*((_BYTE *)v24 + 44) & 7) != 3 )
                  return 0;
                v22 = v24 + 3;
              }
              if ( (*((_BYTE *)v22 + 20) & 8) != 0 )
                return 0;
              v23 = 1;
            }
            v18 = (unsigned int)(v18 + 1);
            if ( v27 == (_DWORD)v18 )
            {
              if ( v23 )
                goto LABEL_7;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v15[3] == sub_C33340() )
    {
      v17 = (void **)v26[4];
      if ( (*((_BYTE *)v17 + 20) & 7) != 3 )
        return 0;
    }
    else
    {
      v17 = v26 + 3;
      if ( (*((_BYTE *)v26 + 44) & 7) != 3 )
        return 0;
    }
    if ( (*((_BYTE *)v17 + 20) & 8) != 0 )
      return 0;
  }
LABEL_7:
  if ( *a4 != 47 )
    return 0;
  v12 = (__int64 *)*((_QWORD *)a4 - 8);
  v13 = (__int64 *)*((_QWORD *)a4 - 4);
  if ( a2 != v12 || !v13 )
  {
    if ( !v12 || a2 != v13 )
      return 0;
    v13 = (__int64 *)*((_QWORD *)a4 - 8);
  }
  v14 = -9;
  if ( a4[1] >> 1 != 127 )
    v14 = (a4[1] >> 1) & 0x77;
  return sub_117CA00(a1, v13, v14 | (8 * a6), a5);
}
