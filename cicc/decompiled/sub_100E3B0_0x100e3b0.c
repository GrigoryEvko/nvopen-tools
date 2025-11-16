// Function: sub_100E3B0
// Address: 0x100e3b0
//
bool __fastcall sub_100E3B0(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  bool v7; // al
  __int64 v8; // r15
  __int64 v9; // rdx
  void **v10; // rax
  void **v11; // r14
  void **v12; // r14
  int v13; // r14d
  unsigned int v14; // r15d
  void **v15; // rax
  void **v16; // rdx
  char v17; // al
  _BYTE *v18; // rdx
  char v19; // [rsp-41h] [rbp-41h]
  void **v20; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 18 )
  {
    if ( *(void **)(v5 + 24) == sub_C33340() )
      v6 = *(_QWORD *)(v5 + 32);
    else
      v6 = v5 + 24;
    v7 = (*(_BYTE *)(v6 + 20) & 7) == 3;
  }
  else
  {
    v8 = *(_QWORD *)(v5 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v10 = (void **)sub_AD7630(v5, 0, v9);
    v11 = v10;
    if ( !v10 || *(_BYTE *)v10 != 18 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v13 = *(_DWORD *)(v8 + 32);
        if ( v13 )
        {
          v19 = 0;
          v14 = 0;
          while ( 1 )
          {
            v15 = (void **)sub_AD69F0((unsigned __int8 *)v5, v14);
            v16 = v15;
            if ( !v15 )
              break;
            v17 = *(_BYTE *)v15;
            v20 = v16;
            if ( v17 != 13 )
            {
              if ( v17 != 18 )
                return 0;
              v18 = v16[3] == sub_C33340() ? v20[4] : v20 + 3;
              if ( (v18[20] & 7) != 3 )
                return 0;
              v19 = 1;
            }
            if ( v13 == ++v14 )
            {
              if ( v19 )
                goto LABEL_8;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v10[3] == sub_C33340() )
      v12 = (void **)v11[4];
    else
      v12 = v11 + 3;
    v7 = (*((_BYTE *)v12 + 20) & 7) == 3;
  }
  if ( !v7 )
    return 0;
LABEL_8:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  return *((_QWORD *)a3 - 4) == *(_QWORD *)(a1 + 8);
}
