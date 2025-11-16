// Function: sub_10A71B0
// Address: 0x10a71b0
//
__int64 __fastcall sub_10A71B0(_QWORD **a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  unsigned int v7; // r13d
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  unsigned int v11; // r13d
  bool v12; // al
  __int64 *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdx
  _BYTE *v16; // rax
  unsigned int v17; // r13d
  __int64 v18; // r13
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // r13d
  bool v22; // al
  bool v23; // r13
  unsigned int v24; // r15d
  __int64 v25; // rax
  unsigned int v26; // r13d
  bool v27; // r13
  unsigned int v28; // r15d
  __int64 v29; // rax
  unsigned int v30; // r13d
  int v31; // [rsp-3Ch] [rbp-3Ch]
  int v32; // [rsp-3Ch] [rbp-3Ch]

  if ( !a2 )
    return 0;
  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 != 42 )
    goto LABEL_3;
  v9 = *((_QWORD *)v2 - 8);
  if ( !v9 )
    goto LABEL_3;
  **a1 = v9;
  v10 = *((_QWORD *)v2 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    v12 = v11 <= 0x40 ? *(_QWORD *)(v10 + 24) == 1 : v11 - 1 == (unsigned int)sub_C444A0(v10 + 24);
  }
  else
  {
    v14 = *(_QWORD *)(v10 + 8);
    v15 = (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17;
    if ( (unsigned int)v15 > 1 || *(_BYTE *)v10 > 0x15u )
      goto LABEL_3;
    v16 = sub_AD7630(v10, 0, v15);
    if ( !v16 || *v16 != 17 )
    {
      if ( *(_BYTE *)(v14 + 8) == 17 )
      {
        v31 = *(_DWORD *)(v14 + 32);
        if ( v31 )
        {
          v23 = 0;
          v24 = 0;
          while ( 1 )
          {
            v25 = sub_AD69F0((unsigned __int8 *)v10, v24);
            if ( !v25 )
              break;
            if ( *(_BYTE *)v25 != 13 )
            {
              if ( *(_BYTE *)v25 != 17 )
                break;
              v26 = *(_DWORD *)(v25 + 32);
              v23 = v26 <= 0x40 ? *(_QWORD *)(v25 + 24) == 1 : v26 - 1 == (unsigned int)sub_C444A0(v25 + 24);
              if ( !v23 )
                break;
            }
            if ( v31 == ++v24 )
            {
              if ( v23 )
                goto LABEL_19;
              goto LABEL_3;
            }
          }
        }
      }
      goto LABEL_3;
    }
    v17 = *((_DWORD *)v16 + 8);
    v12 = v17 <= 0x40 ? *((_QWORD *)v16 + 3) == 1 : v17 - 1 == (unsigned int)sub_C444A0((__int64)(v16 + 24));
  }
  if ( !v12 )
    goto LABEL_3;
LABEL_19:
  v13 = a1[1];
  if ( v13 )
    *v13 = v10;
  result = sub_996420(a1 + 2, 30, *(unsigned __int8 **)(a2 - 32));
  if ( !(_BYTE)result )
  {
LABEL_3:
    v3 = *(_BYTE **)(a2 - 32);
    if ( *v3 == 42 )
    {
      v5 = *((_QWORD *)v3 - 8);
      if ( v5 )
      {
        **a1 = v5;
        v6 = *((_QWORD *)v3 - 4);
        if ( *(_BYTE *)v6 == 17 )
        {
          v7 = *(_DWORD *)(v6 + 32);
          if ( v7 <= 0x40 )
          {
            if ( *(_QWORD *)(v6 + 24) == 1 )
              goto LABEL_11;
          }
          else if ( (unsigned int)sub_C444A0(v6 + 24) == v7 - 1 )
          {
LABEL_11:
            v8 = a1[1];
            if ( v8 )
              *v8 = v6;
            return sub_996420(a1 + 2, 30, *(unsigned __int8 **)(a2 - 64));
          }
        }
        else
        {
          v18 = *(_QWORD *)(v6 + 8);
          v19 = (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17;
          if ( (unsigned int)v19 <= 1 && *(_BYTE *)v6 <= 0x15u )
          {
            v20 = sub_AD7630(v6, 0, v19);
            if ( v20 && *v20 == 17 )
            {
              v21 = *((_DWORD *)v20 + 8);
              if ( v21 <= 0x40 )
                v22 = *((_QWORD *)v20 + 3) == 1;
              else
                v22 = v21 - 1 == (unsigned int)sub_C444A0((__int64)(v20 + 24));
              if ( v22 )
                goto LABEL_11;
            }
            else if ( *(_BYTE *)(v18 + 8) == 17 )
            {
              v32 = *(_DWORD *)(v18 + 32);
              if ( v32 )
              {
                v27 = 0;
                v28 = 0;
                while ( 1 )
                {
                  v29 = sub_AD69F0((unsigned __int8 *)v6, v28);
                  if ( !v29 )
                    break;
                  if ( *(_BYTE *)v29 != 13 )
                  {
                    if ( *(_BYTE *)v29 != 17 )
                      break;
                    v30 = *(_DWORD *)(v29 + 32);
                    v27 = v30 <= 0x40 ? *(_QWORD *)(v29 + 24) == 1 : v30 - 1 == (unsigned int)sub_C444A0(v29 + 24);
                    if ( !v27 )
                      break;
                  }
                  if ( v32 == ++v28 )
                  {
                    if ( v27 )
                      goto LABEL_11;
                    return 0;
                  }
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  return result;
}
