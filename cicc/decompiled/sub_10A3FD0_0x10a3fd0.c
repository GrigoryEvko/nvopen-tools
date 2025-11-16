// Function: sub_10A3FD0
// Address: 0x10a3fd0
//
__int64 __fastcall sub_10A3FD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int8 *v10; // r14
  __int64 v11; // rdx
  unsigned int v12; // r13d
  bool v13; // al
  _QWORD *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r13
  _BYTE *v29; // rax
  unsigned int v30; // r13d
  int v31; // eax
  unsigned int v32; // r15d
  char v33; // r13
  __int64 v34; // rax
  unsigned int v35; // r13d
  int v36; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)a2 != 68 )
    goto LABEL_2;
  v15 = *(_QWORD *)(a2 - 32);
  v16 = *(_QWORD *)(v15 + 16);
  if ( !v16 )
    goto LABEL_2;
  if ( *(_QWORD *)(v16 + 8) )
    goto LABEL_2;
  if ( *(_BYTE *)v15 != 59 )
    goto LABEL_2;
  v17 = *(_QWORD *)(v15 - 64);
  v18 = *(_QWORD *)(v17 + 16);
  if ( !v18 || *(_QWORD *)(v18 + 8) )
    goto LABEL_2;
  if ( *(_BYTE *)v17 == 67 && (v25 = *(_QWORD *)(v17 - 32), (v26 = *(_QWORD *)(v25 + 16)) != 0) && !*(_QWORD *)(v26 + 8) )
  {
    if ( (unsigned __int8)sub_109D4A0(a1, v25) )
      goto LABEL_31;
    v27 = *(_QWORD *)(v17 + 16);
    if ( !v27 )
      goto LABEL_2;
    v19 = a1 + 40;
    if ( *(_QWORD *)(v27 + 8) )
      goto LABEL_2;
  }
  else
  {
    v19 = a1 + 40;
  }
  if ( !(unsigned __int8)sub_109D4A0(v19, v17) )
    goto LABEL_2;
LABEL_31:
  result = sub_991580(a1 + 80, *(_QWORD *)(v15 - 32));
  if ( !(_BYTE)result )
  {
LABEL_2:
    v4 = *(_QWORD *)(a2 + 16);
    if ( v4 )
    {
      if ( !*(_QWORD *)(v4 + 8) && *(_BYTE *)a2 == 59 )
      {
        v6 = *(_QWORD *)(a2 - 64);
        v7 = *(_QWORD *)(v6 + 16);
        if ( v7 )
        {
          if ( !*(_QWORD *)(v7 + 8) )
          {
            if ( *(_BYTE *)v6 != 67 )
              goto LABEL_10;
            v20 = *(_QWORD *)(v6 - 32);
            v21 = *(_QWORD *)(v20 + 16);
            if ( !v21 )
              goto LABEL_10;
            if ( *(_QWORD *)(v21 + 8) )
              goto LABEL_10;
            if ( *(_BYTE *)v20 != 85 )
              goto LABEL_10;
            v22 = *(_QWORD *)(v20 - 32);
            if ( !v22 )
              goto LABEL_10;
            if ( *(_BYTE *)v22 )
              goto LABEL_10;
            if ( *(_QWORD *)(v22 + 24) != *(_QWORD *)(v20 + 80) )
              goto LABEL_10;
            if ( *(_DWORD *)(v22 + 36) != *(_DWORD *)(a1 + 96) )
              goto LABEL_10;
            v23 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
            if ( **(_QWORD **)(a1 + 112) != *(_QWORD *)(v20 + 32 * (*(unsigned int *)(a1 + 104) - v23)) )
              goto LABEL_10;
            if ( (unsigned __int8)sub_993A50(
                                    (_QWORD **)(a1 + 128),
                                    *(_QWORD *)(v20 + 32 * (*(unsigned int *)(a1 + 120) - v23))) )
              return sub_991580(a1 + 176, *(_QWORD *)(a2 - 32));
            v24 = *(_QWORD *)(v6 + 16);
            if ( v24 )
            {
              if ( !*(_QWORD *)(v24 + 8) )
              {
LABEL_10:
                if ( *(_BYTE *)v6 == 85 )
                {
                  v8 = *(_QWORD *)(v6 - 32);
                  if ( v8 )
                  {
                    if ( !*(_BYTE *)v8
                      && *(_QWORD *)(v8 + 24) == *(_QWORD *)(v6 + 80)
                      && *(_DWORD *)(v8 + 36) == *(_DWORD *)(a1 + 136) )
                    {
                      v9 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
                      if ( **(_QWORD **)(a1 + 152) == *(_QWORD *)(v6 + 32 * (*(unsigned int *)(a1 + 144) - v9)) )
                      {
                        v10 = *(unsigned __int8 **)(v6 + 32 * (*(unsigned int *)(a1 + 160) - v9));
                        v11 = *v10;
                        if ( (_BYTE)v11 == 17 )
                        {
                          v12 = *((_DWORD *)v10 + 8);
                          if ( v12 <= 0x40 )
                            v13 = *((_QWORD *)v10 + 3) == 1;
                          else
                            v13 = v12 - 1 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
                          if ( v13 )
                            goto LABEL_20;
                          return 0;
                        }
                        v28 = *((_QWORD *)v10 + 1);
                        if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 <= 1 && (unsigned __int8)v11 <= 0x15u )
                        {
                          v29 = sub_AD7630((__int64)v10, 0, v11);
                          if ( !v29 || *v29 != 17 )
                          {
                            if ( *(_BYTE *)(v28 + 8) == 17 )
                            {
                              v31 = *(_DWORD *)(v28 + 32);
                              v32 = 0;
                              v33 = 0;
                              v36 = v31;
                              while ( v36 != v32 )
                              {
                                v34 = sub_AD69F0(v10, v32);
                                if ( !v34 )
                                  return 0;
                                if ( *(_BYTE *)v34 != 13 )
                                {
                                  if ( *(_BYTE *)v34 != 17 )
                                    return 0;
                                  v35 = *(_DWORD *)(v34 + 32);
                                  if ( v35 <= 0x40 )
                                  {
                                    if ( *(_QWORD *)(v34 + 24) != 1 )
                                      return 0;
                                  }
                                  else if ( (unsigned int)sub_C444A0(v34 + 24) != v35 - 1 )
                                  {
                                    return 0;
                                  }
                                  v33 = 1;
                                }
                                ++v32;
                              }
                              if ( v33 )
                                goto LABEL_20;
                            }
                            return 0;
                          }
                          v30 = *((_DWORD *)v29 + 8);
                          if ( v30 <= 0x40 )
                          {
                            if ( *((_QWORD *)v29 + 3) != 1 )
                              return 0;
                          }
                          else if ( (unsigned int)sub_C444A0((__int64)(v29 + 24)) != v30 - 1 )
                          {
                            return 0;
                          }
LABEL_20:
                          v14 = *(_QWORD **)(a1 + 168);
                          if ( v14 )
                            *v14 = v10;
                          return sub_991580(a1 + 176, *(_QWORD *)(a2 - 32));
                        }
                      }
                    }
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
