// Function: sub_2E4F9C0
// Address: 0x2e4f9c0
//
__int64 __fastcall sub_2E4F9C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, _BYTE *a6)
{
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // r11
  __int64 v13; // r15
  _QWORD *v14; // r14
  unsigned int v16; // r13d
  unsigned int v17; // ecx
  __int64 v18; // rax
  int v19; // r13d
  __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r11
  int v23; // edi
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // r10
  __int64 v27; // r9
  _DWORD *v28; // rdx
  __int64 v29; // r9
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+38h] [rbp-38h]

  v7 = a3;
  v8 = *(_QWORD *)(a2 + 24);
  v9 = *(_QWORD *)(a3 + 24);
  v33 = v8;
  if ( v9 == v8 )
    goto LABEL_14;
  if ( *(_DWORD *)(v9 + 72) != 1 || **(_QWORD **)(v9 + 64) != v8 )
    return 0;
  v11 = *((unsigned int *)a5 + 2);
  if ( !(_DWORD)v11 )
  {
LABEL_14:
    if ( (*(_BYTE *)a2 & 4) == 0 )
    {
      while ( (*(_BYTE *)(a2 + 44) & 8) != 0 )
        a2 = *(_QWORD *)(a2 + 8);
    }
    v18 = *(_QWORD *)(a2 + 8);
    v19 = *(_DWORD *)(a1 + 72);
    v20 = v33 + 48;
    if ( v19 )
    {
      while ( 1 )
      {
LABEL_16:
        if ( v7 == v18 )
        {
LABEL_23:
          if ( v18 != v20 )
            return 1;
        }
        else
        {
          while ( v18 != v20 )
          {
            if ( (unsigned __int16)(*(_WORD *)(v18 + 68) - 14) > 4u )
            {
              if ( v7 == v18 )
                return 1;
              v21 = *(_QWORD *)(v18 + 32);
              v22 = v21 + 40LL * (*(_DWORD *)(v18 + 40) & 0xFFFFFF);
              if ( v21 != v22 )
              {
                while ( *(_BYTE *)v21 != 12 )
                {
                  if ( !*(_BYTE *)v21 && (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
                  {
                    v23 = *(_DWORD *)(v21 + 8);
                    if ( v23 >= 0 )
                    {
                      if ( *(_QWORD *)(a4 + 88) )
                      {
                        v24 = *(_QWORD *)(a4 + 64);
                        if ( v24 )
                        {
                          v25 = a4 + 56;
                          do
                          {
                            while ( 1 )
                            {
                              v26 = *(_QWORD *)(v24 + 16);
                              v27 = *(_QWORD *)(v24 + 24);
                              if ( (unsigned int)v23 <= *(_DWORD *)(v24 + 32) )
                                break;
                              v24 = *(_QWORD *)(v24 + 24);
                              if ( !v27 )
                                goto LABEL_43;
                            }
                            v25 = v24;
                            v24 = *(_QWORD *)(v24 + 16);
                          }
                          while ( v26 );
LABEL_43:
                          if ( a4 + 56 != v25 && (unsigned int)v23 >= *(_DWORD *)(v25 + 32) )
                            return 0;
                        }
                      }
                      else
                      {
                        v28 = *(_DWORD **)a4;
                        v29 = *(_QWORD *)a4 + 4LL * *(unsigned int *)(a4 + 8);
                        if ( *(_QWORD *)a4 != v29 )
                        {
                          while ( v23 != *v28 )
                          {
                            if ( (_DWORD *)v29 == ++v28 )
                              goto LABEL_45;
                          }
                          if ( (_DWORD *)v29 != v28 )
                            return 0;
                        }
                      }
                    }
                  }
LABEL_45:
                  v21 += 40;
                  if ( v22 == v21 )
                    goto LABEL_46;
                }
                return 0;
              }
LABEL_46:
              --v19;
              if ( (*(_BYTE *)v18 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v18 + 44) & 8) != 0 )
                  v18 = *(_QWORD *)(v18 + 8);
              }
              v18 = *(_QWORD *)(v18 + 8);
              if ( !v19 )
                return 0;
              goto LABEL_16;
            }
            if ( (*(_BYTE *)v18 & 4) != 0 )
            {
              v18 = *(_QWORD *)(v18 + 8);
              if ( v7 == v18 )
                goto LABEL_23;
            }
            else
            {
              while ( (*(_BYTE *)(v18 + 44) & 8) != 0 )
                v18 = *(_QWORD *)(v18 + 8);
              v18 = *(_QWORD *)(v18 + 8);
              if ( v7 == v18 )
                goto LABEL_23;
            }
          }
        }
        v20 = v9 + 48;
        *a6 = 1;
        v18 = *(_QWORD *)(v9 + 56);
      }
    }
    return 0;
  }
  v32 = *(_QWORD *)(a3 + 24);
  v31 = a2;
  v12 = *a5;
  v13 = 0;
  v36 = 8 * v11;
  v14 = *(_QWORD **)(a1 + 24);
  while ( 1 )
  {
    v16 = *(_DWORD *)(v12 + v13 + 4);
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*v14 + 16LL)
                                                                                       + 200LL))(
                                              *(_QWORD *)(*v14 + 16LL),
                                              v7)
                                          + 248)
                              + 16LL)
                  + v16) )
    {
      result = *(_QWORD *)(v14[48] + 8LL * (v16 >> 6)) & (1LL << v16);
      if ( !result )
        return result;
    }
    v12 = *a5;
    v14 = *(_QWORD **)(a1 + 24);
    v17 = *(_DWORD *)(*a5 + v13 + 4);
    v7 = v17 >> 6;
    if ( (*(_QWORD *)(v14[48] + 8 * v7) & (1LL << v17)) != 0 )
      return 0;
    v13 += 8;
    if ( v36 == v13 )
    {
      v9 = v32;
      a2 = v31;
      v7 = a3;
      goto LABEL_14;
    }
  }
}
