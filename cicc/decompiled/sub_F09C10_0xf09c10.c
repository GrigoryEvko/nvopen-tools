// Function: sub_F09C10
// Address: 0xf09c10
//
__int64 __fastcall sub_F09C10(int a1, unsigned __int8 *a2, _QWORD *a3, _QWORD *a4, _BYTE *a5)
{
  unsigned __int8 *v6; // r13
  int v7; // eax
  unsigned __int8 *v9; // r12
  __int64 v10; // rdx
  unsigned int v11; // edi
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned __int8 *v14; // rax
  __int64 v15; // r13
  _BYTE *v16; // rax
  unsigned int v17; // edi
  int v18; // r13d
  char v19; // r14
  unsigned int v20; // r15d
  __int64 v21; // rax
  unsigned int v22; // edi
  __int64 v23; // rsi

  *a3 = *((_QWORD *)a2 - 8);
  v6 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  *a4 = v6;
  v7 = *a2;
  if ( (a1 & 0xFFFFFFFD) != 0xD )
  {
    if ( (unsigned int)(a1 - 28) > 2 || !a5 || *a5 != 56 || (_BYTE)v7 != 55 )
      return (unsigned int)(v7 - 29);
    v9 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v10 = *v9;
    if ( (_BYTE)v10 == 17 )
    {
      v11 = *((_DWORD *)v9 + 8);
      v12 = *((_QWORD *)v9 + 3);
      v13 = 1LL << ((unsigned __int8)v11 - 1);
      if ( v11 > 0x40 )
        v12 = *(_QWORD *)(v12 + 8LL * ((v11 - 1) >> 6));
    }
    else
    {
      v15 = *((_QWORD *)v9 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 > 1 || (unsigned __int8)v10 > 0x15u )
        return (unsigned int)(v7 - 29);
      v16 = sub_AD7630((__int64)v9, 0, v10);
      if ( !v16 || *v16 != 17 )
      {
        if ( *(_BYTE *)(v15 + 8) == 17 )
        {
          v18 = *(_DWORD *)(v15 + 32);
          if ( v18 )
          {
            v19 = 0;
            v20 = 0;
            while ( 1 )
            {
              v21 = sub_AD69F0(v9, v20);
              if ( !v21 )
                break;
              if ( *(_BYTE *)v21 != 13 )
              {
                if ( *(_BYTE *)v21 != 17 )
                  break;
                v22 = *(_DWORD *)(v21 + 32);
                v23 = *(_QWORD *)(v21 + 24);
                if ( v22 > 0x40 )
                  v23 = *(_QWORD *)(v23 + 8LL * ((v22 - 1) >> 6));
                if ( (v23 & (1LL << ((unsigned __int8)v22 - 1))) != 0 )
                  break;
                v19 = 1;
              }
              if ( v18 == ++v20 )
              {
                if ( v19 )
                  return 27;
                break;
              }
            }
          }
        }
LABEL_10:
        v7 = *a2;
        return (unsigned int)(v7 - 29);
      }
      v17 = *((_DWORD *)v16 + 8);
      v13 = *((_QWORD *)v16 + 3);
      v12 = 1LL << ((unsigned __int8)v17 - 1);
      if ( v17 > 0x40 )
        v13 = *(_QWORD *)(v13 + 8LL * ((v17 - 1) >> 6));
    }
    if ( (v13 & v12) == 0 )
      return 27;
    goto LABEL_10;
  }
  if ( (_BYTE)v7 != 54 || *v6 > 0x15u || *v6 == 5 )
    return (unsigned int)(v7 - 29);
  if ( (unsigned __int8)sub_AD6CA0((__int64)v6) )
    goto LABEL_10;
  v14 = (unsigned __int8 *)sub_AD64C0(*((_QWORD *)a2 + 1), 1, 0);
  *a4 = sub_AABE40(0x19u, v14, v6);
  return 17;
}
