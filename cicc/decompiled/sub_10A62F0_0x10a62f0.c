// Function: sub_10A62F0
// Address: 0x10a62f0
//
__int64 __fastcall sub_10A62F0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // al
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rdx
  void **v7; // rax
  void **v8; // r13
  void **v9; // r13
  int v10; // r14d
  unsigned int v11; // r15d
  void **v12; // rax
  void **v13; // r13
  char v14; // al
  _BYTE *v15; // r13
  char v16; // [rsp+Fh] [rbp-31h]

  if ( *(_BYTE *)a2 == 18 )
  {
    if ( *(void **)(a2 + 24) == sub_C33340() )
      v2 = *(_QWORD *)(a2 + 32);
    else
      v2 = a2 + 24;
    v3 = (*(_BYTE *)(v2 + 20) & 7) == 3;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17;
    if ( (unsigned int)v6 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    v7 = (void **)sub_AD7630(a2, 0, v6);
    v8 = v7;
    if ( !v7 || *(_BYTE *)v7 != 18 )
    {
      if ( *(_BYTE *)(v5 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v5 + 32);
        if ( v10 )
        {
          v16 = 0;
          v11 = 0;
          while ( 1 )
          {
            v12 = (void **)sub_AD69F0((unsigned __int8 *)a2, v11);
            v13 = v12;
            if ( !v12 )
              break;
            v14 = *(_BYTE *)v12;
            if ( v14 != 13 )
            {
              if ( v14 != 18 )
                return 0;
              v15 = v13[3] == sub_C33340() ? v13[4] : v13 + 3;
              if ( (v15[20] & 7) != 3 )
                return 0;
              v16 = 1;
            }
            if ( v10 == ++v11 )
            {
              if ( v16 )
                goto LABEL_6;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v7[3] == sub_C33340() )
      v9 = (void **)v8[4];
    else
      v9 = v8 + 3;
    v3 = (*((_BYTE *)v9 + 20) & 7) == 3;
  }
  if ( !v3 )
    return 0;
LABEL_6:
  result = 1;
  if ( *a1 )
    **a1 = a2;
  return result;
}
