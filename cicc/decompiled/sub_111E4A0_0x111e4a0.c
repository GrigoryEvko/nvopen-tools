// Function: sub_111E4A0
// Address: 0x111e4a0
//
__int64 __fastcall sub_111E4A0(__int64 **a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned int v10; // esi
  int v11; // r13d
  char v12; // r14
  unsigned int v13; // r15d
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rax

  if ( *(_BYTE *)a2 != 17 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17;
    if ( (unsigned int)v8 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v9 = sub_AD7630(a2, 0, v8);
    if ( !v9 || *v9 != 17 )
    {
      if ( *(_BYTE *)(v7 + 8) == 17 )
      {
        v11 = *(_DWORD *)(v7 + 32);
        if ( v11 )
        {
          v12 = 0;
          v13 = 0;
          while ( 1 )
          {
            v14 = sub_AD69F0((unsigned __int8 *)a2, v13);
            if ( !v14 )
              break;
            if ( *(_BYTE *)v14 != 13 )
            {
              if ( *(_BYTE *)v14 != 17 )
                return 0;
              v15 = *(_DWORD *)(v14 + 32);
              v16 = *(_QWORD *)(v14 + 24);
              if ( v15 > 0x40 )
                v16 = *(_QWORD *)(v16 + 8LL * ((v15 - 1) >> 6));
              if ( (v16 & (1LL << ((unsigned __int8)v15 - 1))) != 0 )
                return 0;
              v12 = 1;
            }
            if ( v11 == ++v13 )
            {
              if ( v12 )
                goto LABEL_4;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v10 = *((_DWORD *)v9 + 8);
    v5 = *((_QWORD *)v9 + 3);
    v4 = 1LL << ((unsigned __int8)v10 - 1);
    if ( v10 > 0x40 )
      v5 = *(_QWORD *)(v5 + 8LL * ((v10 - 1) >> 6));
    goto LABEL_3;
  }
  v3 = *(_DWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 24);
  v5 = 1LL << ((unsigned __int8)v3 - 1);
  if ( v3 <= 0x40 )
  {
LABEL_3:
    if ( (v5 & v4) == 0 )
      goto LABEL_4;
    return 0;
  }
  if ( (v5 & *(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6))) != 0 )
    return 0;
LABEL_4:
  result = 1;
  if ( *a1 )
    **a1 = a2;
  return result;
}
