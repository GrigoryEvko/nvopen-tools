// Function: sub_1110BF0
// Address: 0x1110bf0
//
__int64 __fastcall sub_1110BF0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  bool v5; // al
  __int64 *v6; // rdx
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // rdx
  void **v10; // rax
  void **v11; // r13
  void **v12; // r13
  int v13; // r14d
  unsigned int v14; // r15d
  void **v15; // rax
  void **v16; // r13
  char v17; // al
  _BYTE *v18; // r13
  char v19; // [rsp-39h] [rbp-39h]

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 == 18 )
  {
    if ( *(void **)(v3 + 24) == sub_C33340() )
      v4 = *(_QWORD *)(v3 + 32);
    else
      v4 = v3 + 24;
    v5 = (*(_BYTE *)(v4 + 20) & 7) == 3;
  }
  else
  {
    v8 = *(_QWORD *)(v3 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v3 > 0x15u )
      return 0;
    v10 = (void **)sub_AD7630(v3, 0, v9);
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
            v15 = (void **)sub_AD69F0((unsigned __int8 *)v3, v14);
            v16 = v15;
            if ( !v15 )
              break;
            v17 = *(_BYTE *)v15;
            if ( v17 != 13 )
            {
              if ( v17 != 18 )
                return 0;
              v18 = v16[3] == sub_C33340() ? v16[4] : v16 + 3;
              if ( (v18[20] & 7) != 3 )
                return 0;
              v19 = 1;
            }
            if ( v13 == ++v14 )
            {
              if ( v19 )
                goto LABEL_7;
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
    v5 = (*((_BYTE *)v12 + 20) & 7) == 3;
  }
  if ( !v5 )
    return 0;
LABEL_7:
  v6 = a1[1];
  result = 1;
  if ( v6 )
    *v6 = v3;
  return result;
}
