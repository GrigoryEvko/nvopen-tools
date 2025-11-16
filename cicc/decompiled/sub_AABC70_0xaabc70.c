// Function: sub_AABC70
// Address: 0xaabc70
//
__int64 __fastcall sub_AABC70(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r13
  char v13; // al
  __int64 v14; // r13
  int v15; // r14d
  unsigned int v16; // r15d
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BYTE *v21; // r13
  char v22; // al
  _BYTE *v23; // r13
  char v24; // [rsp+Fh] [rbp-31h]

  if ( *(_BYTE *)a2 != 18 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    {
      v8 = sub_AD7630(a2, 0);
      v12 = v8;
      if ( !v8 || *(_BYTE *)v8 != 18 )
      {
        if ( *(_BYTE *)(v7 + 8) == 17 )
        {
          v15 = *(_DWORD *)(v7 + 32);
          if ( v15 )
          {
            v24 = 0;
            v16 = 0;
            while ( 1 )
            {
              v17 = (_BYTE *)sub_AD69F0(a2, v16);
              v21 = v17;
              if ( !v17 )
                break;
              v22 = *v17;
              if ( v22 != 13 )
              {
                if ( v22 != 18 )
                  return 0;
                if ( *((_QWORD *)v21 + 3) == sub_C33340(a2, v16, v18, v19, v20) )
                {
                  v23 = (_BYTE *)*((_QWORD *)v21 + 4);
                  if ( (v23[20] & 7) != 3 )
                    return 0;
                }
                else
                {
                  if ( (v21[44] & 7) != 3 )
                    return 0;
                  v23 = v21 + 24;
                }
                if ( (v23[20] & 8) == 0 )
                  return 0;
                v24 = 1;
              }
              if ( v15 == ++v16 )
              {
                if ( v24 )
                  goto LABEL_7;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      if ( *(_QWORD *)(v8 + 24) == sub_C33340(a2, 0, v9, v10, v11) )
      {
        v14 = *(_QWORD *)(v12 + 32);
        if ( (*(_BYTE *)(v14 + 20) & 7) != 3 )
          return 0;
      }
      else
      {
        v13 = *(_BYTE *)(v12 + 44);
        v14 = v12 + 24;
        if ( (v13 & 7) != 3 )
          return 0;
      }
      if ( (*(_BYTE *)(v14 + 20) & 8) != 0 )
        goto LABEL_7;
    }
    return 0;
  }
  if ( *(_QWORD *)(a2 + 24) != sub_C33340(a1, a2, a3, a4, a5) )
  {
    if ( (*(_BYTE *)(a2 + 44) & 7) == 3 && (*(_BYTE *)(a2 + 44) & 8) != 0 )
      goto LABEL_7;
    return 0;
  }
  v6 = *(_QWORD *)(a2 + 32);
  if ( (*(_BYTE *)(v6 + 20) & 7) != 3 || (*(_BYTE *)(v6 + 20) & 8) == 0 )
    return 0;
LABEL_7:
  if ( *a1 )
    **a1 = a2;
  return 1;
}
