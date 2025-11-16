// Function: sub_10C4F30
// Address: 0x10c4f30
//
__int64 __fastcall sub_10C4F30(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // esi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // r13
  _BYTE *v9; // rax
  int v10; // r13d
  char v11; // r14
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rax
  unsigned int v16; // esi

  if ( *(_BYTE *)a2 != 17 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 )
      return 0;
    v9 = sub_AD7630(a2, 0, a3);
    if ( !v9 || *v9 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v8 + 32);
        if ( v10 )
        {
          v11 = 0;
          v12 = 0;
          while ( 1 )
          {
            v13 = sub_AD69F0((unsigned __int8 *)a2, v12);
            if ( !v13 )
              break;
            if ( *(_BYTE *)v13 != 13 )
            {
              if ( *(_BYTE *)v13 != 17 )
                return 0;
              v14 = *(_DWORD *)(v13 + 32);
              v15 = *(_QWORD *)(v13 + 24);
              if ( v14 > 0x40 )
                v15 = *(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6));
              if ( (v15 & (1LL << ((unsigned __int8)v14 - 1))) == 0 )
                return 0;
              v11 = 1;
            }
            if ( v10 == ++v12 )
            {
              if ( v11 )
                goto LABEL_4;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v16 = *((_DWORD *)v9 + 8);
    v6 = *((_QWORD *)v9 + 3);
    v5 = 1LL << ((unsigned __int8)v16 - 1);
    if ( v16 > 0x40 )
      v6 = *(_QWORD *)(v6 + 8LL * ((v16 - 1) >> 6));
LABEL_3:
    if ( (v6 & v5) != 0 )
      goto LABEL_4;
    return 0;
  }
  v4 = *(_DWORD *)(a2 + 32);
  v5 = *(_QWORD *)(a2 + 24);
  v6 = 1LL << ((unsigned __int8)v4 - 1);
  if ( v4 <= 0x40 )
    goto LABEL_3;
  if ( (v6 & *(_QWORD *)(v5 + 8LL * ((v4 - 1) >> 6))) == 0 )
    return 0;
LABEL_4:
  result = 1;
  if ( *a1 )
    **a1 = a2;
  return result;
}
