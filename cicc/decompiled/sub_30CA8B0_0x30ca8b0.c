// Function: sub_30CA8B0
// Address: 0x30ca8b0
//
__int64 __fastcall sub_30CA8B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r15
  __int64 v3; // r10
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 i; // r12
  char *v7; // rax
  char v8; // al
  __int64 result; // rax
  signed int v10; // r14d
  signed int v11; // r9d
  int v12; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  bool v15; // al
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // rcx
  int v19; // edi
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  int v24; // [rsp+10h] [rbp-50h]
  int v25; // [rsp+14h] [rbp-4Ch]
  __int64 v26; // [rsp+18h] [rbp-48h]
  signed int v27; // [rsp+20h] [rbp-40h]
  int v28; // [rsp+24h] [rbp-3Ch]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 8);
  v28 = 0;
  v2 = *(_QWORD *)(v1 + 32);
  v24 = 0;
  v29 = v1 + 24;
  v25 = 0;
  if ( v2 != v1 + 24 )
  {
    v3 = 0x8000000000041LL;
    do
    {
      while ( 1 )
      {
        if ( !v2 )
          BUG();
        v4 = *(_QWORD *)(v2 + 24);
        v5 = v2 + 16;
        if ( v2 + 16 != v4 )
        {
          if ( !v4 )
            BUG();
          while ( 1 )
          {
            i = *(_QWORD *)(v4 + 32);
            if ( i != v4 + 24 )
              break;
            v4 = *(_QWORD *)(v4 + 8);
            if ( v5 == v4 )
              goto LABEL_10;
            if ( !v4 )
              BUG();
          }
          if ( v5 != v4 )
          {
            v10 = 0;
            v11 = 0;
            do
            {
              if ( !i )
                BUG();
              v12 = *(unsigned __int8 *)(i - 24);
              if ( (_BYTE)v12 == 32 )
              {
                if ( v10 < (int)(((*(_DWORD *)(i - 20) & 0x7FFFFFFu) >> 1) - 1) )
                  v10 = ((*(_DWORD *)(i - 20) & 0x7FFFFFFu) >> 1) - 1;
              }
              else
              {
                v13 = (unsigned int)(v12 - 34);
                if ( (unsigned __int8)v13 <= 0x33u )
                {
                  if ( _bittest64(&v3, v13) )
                  {
                    v14 = *(_QWORD *)(i - 56);
                    if ( v14 )
                    {
                      if ( !*(_BYTE *)v14 && *(_QWORD *)(v14 + 24) == *(_QWORD *)(i + 56) )
                      {
                        v27 = v11;
                        v26 = *(_QWORD *)(i - 56);
                        v15 = sub_B2FC80(v26);
                        v11 = v27;
                        v3 = 0x8000000000041LL;
                        if ( !v15
                          && (*(_BYTE *)(i - 24) != 85
                           || (v22 = *(_QWORD *)(i - 56)) == 0
                           || *(_BYTE *)v22
                           || *(_QWORD *)(v22 + 24) != *(_QWORD *)(i + 56)
                           || (*(_BYTE *)(v22 + 33) & 0x20) == 0) )
                        {
                          v16 = sub_B2D610(v26, 3);
                          v3 = 0x8000000000041LL;
                          if ( v16 )
                          {
                            v17 = *(_QWORD *)(v26 + 80);
                            v18 = v26 + 72;
                            if ( v17 != v26 + 72 )
                            {
                              v19 = 0;
                              do
                              {
                                while ( 1 )
                                {
                                  if ( !v17 )
                                    BUG();
                                  v20 = *(_QWORD *)(v17 + 32);
                                  if ( v17 + 24 != v20 )
                                    break;
                                  v17 = *(_QWORD *)(v17 + 8);
                                  if ( v18 == v17 )
                                    goto LABEL_50;
                                }
                                v21 = 0;
                                do
                                {
                                  v20 = *(_QWORD *)(v20 + 8);
                                  ++v21;
                                }
                                while ( v17 + 24 != v20 );
                                v17 = *(_QWORD *)(v17 + 8);
                                v19 += v21;
                              }
                              while ( v18 != v17 );
LABEL_50:
                              v24 += v19;
                            }
                          }
                          v11 = v27 + 1;
                        }
                      }
                    }
                  }
                }
              }
              for ( i = *(_QWORD *)(i + 8); i == v4 - 24 + 48; i = *(_QWORD *)(v4 + 32) )
              {
                v4 = *(_QWORD *)(v4 + 8);
                if ( v5 == v4 )
                  goto LABEL_29;
                if ( !v4 )
                  BUG();
              }
            }
            while ( v5 != v4 );
LABEL_29:
            if ( v11 >= v10 && v10 > 1 )
              break;
          }
        }
LABEL_10:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v29 == v2 )
          goto LABEL_11;
      }
      ++v25;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v28 >= v10 )
        v10 = v28;
      v28 = v10;
    }
    while ( v29 != v2 );
  }
LABEL_11:
  *(_DWORD *)(a1 + 156) = qword_502F608 - v24;
  v7 = (char *)sub_C94E20((__int64)qword_4F86350);
  if ( v7 )
    v8 = *v7;
  else
    v8 = qword_4F86350[2];
  if ( v8
    || !(_BYTE)qword_502F528
    || (int)qword_502F448 > v25
    || (result = (unsigned int)qword_502F288, v28 < (int)qword_502F368) )
  {
    result = (unsigned int)qword_502F1A8;
  }
  *(_DWORD *)(a1 + 160) = result;
  return result;
}
