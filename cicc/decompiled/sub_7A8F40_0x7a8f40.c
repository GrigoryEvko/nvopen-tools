// Function: sub_7A8F40
// Address: 0x7a8f40
//
unsigned __int64 __fastcall sub_7A8F40(__int64 a1, __int64 a2, __int64 **a3)
{
  unsigned __int64 result; // rax
  __int64 *v7; // rbx
  unsigned __int64 v8; // r15
  char v9; // al
  __int64 *v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // r15
  unsigned __int64 v20; // r14
  __int64 i; // rbx
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 j; // rdi
  _QWORD v25[7]; // [rsp+18h] [rbp-38h] BYREF

  result = (*a3)[21];
  v7 = *(__int64 **)result;
  if ( a1 )
  {
    v8 = *(_QWORD *)(a1 + 128);
    if ( v7 )
      goto LABEL_5;
LABEL_18:
    result = (unsigned __int64)&dword_4D0425C;
    if ( dword_4D0425C && a1 && (*(_BYTE *)(a1 + 144) & 4) != 0 )
    {
LABEL_24:
      result = (unsigned __int64)&dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        v16 = *a3;
        v17 = **a3;
        if ( !v17 )
          goto LABEL_31;
        if ( *((_BYTE *)v16 + 140) == 12 )
        {
          v18 = *a3;
          do
            v18 = (__int64 *)v18[20];
          while ( *((_BYTE *)v18 + 140) == 12 );
          v17 = *v18;
        }
        result = *(_QWORD *)(v17 + 96);
        if ( *(char *)(result + 178) >= 0 )
        {
LABEL_31:
          result = v16[21];
          v19 = *(__int64 **)result;
          if ( *(_QWORD *)result )
          {
            result = (unsigned __int64)v25;
            v20 = 0;
            do
            {
              if ( (v19[12] & 0x40) != 0 )
              {
                for ( i = v19[5]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                  ;
                if ( (unsigned int)sub_7A80B0(i) )
                {
                  v22 = v19[13];
                  v23 = *(_QWORD *)(i + 128);
                }
                else
                {
                  v22 = v19[13];
                  v23 = *(_QWORD *)(*(_QWORD *)(i + 168) + 32LL);
                }
                result = (unsigned __int64)a3[6] + 1;
                if ( result == v22 + v23 )
                {
                  for ( j = v19[5]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                    ;
                  v25[0] = v22;
                  result = sub_7A6650(j, v25);
                  if ( result )
                  {
                    if ( v20 )
                    {
                      if ( *(_QWORD *)(result + 128) > *(_QWORD *)(v20 + 128) )
                        v20 = result;
                    }
                    else
                    {
                      v20 = result;
                    }
                  }
                }
              }
              v19 = (__int64 *)*v19;
            }
            while ( v19 );
            if ( v20 && (*(_BYTE *)(v20 + 144) & 4) != 0 )
            {
              result = dword_4F06BA0 * (_QWORD)((char *)a3[6] - *(_QWORD *)(v20 + 128) + 1);
              if ( result > *(unsigned __int8 *)(a1 + 137) + *(unsigned __int8 *)(v20 + 136) )
                return sub_684B30(0x4BEu, (_DWORD *)(a1 + 64));
            }
          }
        }
      }
    }
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 104);
    if ( !v7 )
      return result;
LABEL_5:
    while ( 1 )
    {
      v9 = *((_BYTE *)v7 + 96);
      if ( (v9 & 3) != 0 && (v9 & 0x40) != 0 && !(unsigned int)sub_7A80B0(v7[5]) )
      {
        v10 = (__int64 *)v7[5];
        v11 = v10[21];
        v12 = *(_QWORD *)(v11 + 32);
        v13 = *(unsigned int *)(v11 + 40);
        v14 = v12 % v13;
        if ( v12 % v13 && v13 - v14 <= unk_4F06AC0 && v12 <= v14 - v13 + unk_4F06AC0 )
          v12 += v13 - v14;
        v15 = v7[13];
        if ( v15 + v12 > v8 && v15 < v8 )
          break;
      }
      v7 = (__int64 *)*v7;
      if ( !v7 )
        goto LABEL_18;
    }
    if ( !a1 )
      return sub_686C80(0x4C2u, (FILE *)(a2 + 72), **(_QWORD **)(a2 + 40), *v10);
    sub_684B30(0x4BDu, (_DWORD *)(a1 + 64));
    result = dword_4D0425C;
    if ( dword_4D0425C && (*(_BYTE *)(a1 + 144) & 4) != 0 )
      goto LABEL_24;
  }
  return result;
}
