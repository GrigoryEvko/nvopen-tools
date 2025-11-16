// Function: sub_8CC930
// Address: 0x8cc930
//
__int64 __fastcall sub_8CC930(__int64 a1, int a2)
{
  __int64 result; // rax
  unsigned __int8 *v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  int v7; // edx
  int v8; // ecx
  unsigned __int8 v9; // si
  unsigned int v10; // [rsp+Ch] [rbp-34h]

  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u || (*(_BYTE *)(a1 + 177) & 0x84) == 0 )
  {
    result = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      v4 = *(unsigned __int8 **)a1;
      if ( *(_QWORD *)(a1 + 8) )
      {
        if ( v4 )
        {
          result = sub_8C6B40(*(_QWORD *)a1);
          if ( (_DWORD)result )
          {
            v5 = sub_880F80((__int64)v4);
            result = *(_QWORD *)v4;
            v6 = *(_QWORD *)(*(_QWORD *)v4 + 32LL);
            if ( v6 )
            {
              v10 = 0;
              do
              {
                if ( *(_DWORD *)(v6 + 40) == -1 )
                  goto LABEL_14;
                result = sub_880F80(v6);
                if ( v5 == result )
                  goto LABEL_14;
                result = a2 ? sub_8C6230(v6, (__int64)v4) : sub_8C7F70(v6, (__int64)v4);
                if ( !(_DWORD)result )
                  goto LABEL_14;
                if ( !(unsigned int)sub_8C6B40(v6) )
                {
                  result = sub_87D520(v6);
                  if ( result && (*(_BYTE *)(result - 8) & 2) == 0 )
                    *(_BYTE *)(result + 90) |= 8u;
                  goto LABEL_14;
                }
                result = v10;
                if ( !v10 )
                {
                  result = *(unsigned __int8 *)(v6 + 80);
                  v7 = v4[80];
                  v8 = *(unsigned __int8 *)(v6 + 80);
                  if ( (_BYTE)result == (_BYTE)v7 )
                  {
                    result = sub_8CA500(a1, *(_QWORD *)(v6 + 88));
                    v10 = 1;
                    goto LABEL_14;
                  }
                  if ( (v4[81] & 0x10) == 0 )
                  {
                    v9 = result - 4;
                    if ( (unsigned __int8)(v7 - 4) <= 2u )
                    {
                      if ( v9 > 2u )
                        goto LABEL_33;
                      goto LABEL_27;
                    }
                    if ( (_BYTE)v7 == 3 )
                    {
                      if ( v4[104] )
                      {
                        if ( v9 > 2u )
                        {
LABEL_33:
                          if ( (_BYTE)v8 != 3 )
                          {
                            if ( (v8 & 0xFB) != 0x13 )
                            {
LABEL_35:
                              if ( (unsigned __int8)(v8 - 20) <= 1u )
                                goto LABEL_36;
                              result = (unsigned int)(v8 - 7);
                              if ( (((_BYTE)v8 - 7) & 0xFD) != 0 || (_BYTE)v8 != 9 && (_BYTE)v8 != 7 )
                                goto LABEL_71;
                              result = *(_QWORD *)(v6 + 88);
                              if ( !result )
                                goto LABEL_14;
                              if ( (*(_BYTE *)(result + 170) & 0x10) != 0
                                && (result = *(_QWORD *)(result + 216), *(_QWORD *)result) )
                              {
LABEL_36:
                                LOBYTE(result) = v8;
                              }
                              else
                              {
LABEL_71:
                                if ( (_BYTE)v8 != 17 )
                                  goto LABEL_14;
                                result = sub_8780F0(v6);
                                if ( !(_DWORD)result )
                                  goto LABEL_14;
LABEL_64:
                                LOBYTE(result) = *(_BYTE *)(v6 + 80);
                              }
                              goto LABEL_37;
                            }
LABEL_27:
                            result = sub_8C6700((__int64 *)a1, (unsigned int *)(v6 + 48), 0x42Au, 0x425u);
                            goto LABEL_14;
                          }
                          if ( !*(_BYTE *)(v6 + 104) )
                          {
                            if ( (unsigned int)sub_8C6B40(v6) )
                              goto LABEL_64;
                            v8 = *(unsigned __int8 *)(v6 + 80);
                            result = *(unsigned __int8 *)(v6 + 80);
                            if ( (v8 & 0xFB) != 0x13 )
                            {
                              if ( (_BYTE)v8 != 3 )
                                goto LABEL_35;
                              if ( !*(_BYTE *)(v6 + 104) )
                                goto LABEL_14;
                              result = *(_QWORD *)(v6 + 88);
                              if ( (*(_BYTE *)(result + 177) & 0x10) == 0 )
                                goto LABEL_14;
                              result = *(_QWORD *)(result + 168);
                              if ( !*(_QWORD *)(result + 168) )
                                goto LABEL_14;
                              goto LABEL_38;
                            }
LABEL_37:
                            if ( (unsigned __int8)(result - 4) <= 2u )
                              goto LABEL_38;
LABEL_26:
                            if ( (_BYTE)result != 3 || !*(_BYTE *)(v6 + 104) )
                              goto LABEL_27;
LABEL_38:
                            v7 = v4[80];
                          }
                          result = (unsigned int)(v7 - 4);
                          if ( (unsigned __int8)(v7 - 4) <= 2u )
                            goto LABEL_27;
                          if ( (_BYTE)v7 != 3 )
                            goto LABEL_14;
                        }
                        if ( v4[104] )
                          goto LABEL_27;
                      }
                      else if ( v9 > 2u )
                      {
                        goto LABEL_26;
                      }
                      result = sub_8C6B40((__int64)v4);
                      if ( !(_DWORD)result )
                        goto LABEL_14;
                      goto LABEL_27;
                    }
                    if ( v9 > 2u )
                      goto LABEL_26;
                  }
                }
LABEL_14:
                v6 = *(_QWORD *)(v6 + 8);
              }
              while ( v6 );
            }
          }
        }
      }
      if ( !*(_QWORD *)(a1 + 32) )
        return sub_8CA0A0(a1, 1u);
    }
  }
  return result;
}
