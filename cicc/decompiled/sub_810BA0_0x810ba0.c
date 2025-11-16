// Function: sub_810BA0
// Address: 0x810ba0
//
void __fastcall sub_810BA0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  char v6; // r9
  __int64 *v7; // rcx
  char v8; // r9
  __int64 v9; // r8
  __int64 *i; // rdx
  __int64 v11; // rax
  __int64 *v12; // rax
  signed __int64 v13; // r13
  __int64 **v14; // rax
  __int64 v15; // rsi
  __int64 *v16; // rsi
  __int64 v17; // rdx
  _BYTE v18[96]; // [rsp+0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 112);
  v4 = *(_QWORD *)(v3 + 16);
  v5 = v4;
  if ( (*(_BYTE *)(a1 + 96) & 2) == 0 )
    v5 = *(_QWORD *)(v3 + 8);
  do
  {
    sub_810650(*(_QWORD *)(*(_QWORD *)(v4 + 16) + 40LL), 1, a2);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v4 != *(_QWORD *)(v5 + 8) );
  v6 = *(_BYTE *)(a1 + 96);
  if ( (v6 & 4) != 0 )
  {
    v7 = *(__int64 **)(a1 + 40);
    v8 = v6 & 3;
    v9 = -1;
    for ( i = **(__int64 ***)(*(_QWORD *)(a1 + 56) + 168LL); ; i = (__int64 *)*i )
    {
      if ( (i[12] & 4) != 0 )
      {
        v12 = (__int64 *)i[5];
        if ( v12 == v7 || v12 && v7 && dword_4F07588 && (v11 = v12[4], v7[4] == v11) && v11 )
        {
          if ( v8 == 1 && **(_QWORD **)(*(_QWORD *)(a1 + 56) + 168LL) )
          {
            v14 = **(__int64 ****)(*(_QWORD *)(a1 + 56) + 168LL);
            while ( 1 )
            {
              if ( ((_BYTE)v14[12] & 2) != 0 )
              {
                v16 = v14[5];
                if ( v7 == v16 )
                  break;
                if ( v16 )
                {
                  if ( v7 )
                  {
                    if ( dword_4F07588 )
                    {
                      v15 = v16[4];
                      if ( v7[4] == v15 )
                      {
                        if ( v15 )
                          break;
                      }
                    }
                  }
                }
              }
              v14 = (__int64 **)*v14;
              if ( !v14 )
                goto LABEL_15;
            }
            v13 = -1;
          }
          else
          {
LABEL_15:
            v13 = ++v9;
          }
          if ( (__int64 *)a1 == i )
            break;
        }
      }
    }
    if ( v13 )
    {
      *a2 += 3LL;
      sub_8238B0(qword_4F18BE0, "__A", 3);
      if ( v13 >= 0 )
      {
        if ( v13 > 9 )
        {
          v17 = (int)sub_622470(v13, v18);
        }
        else
        {
          v18[1] = 0;
          v17 = 1;
          v18[0] = v13 + 48;
        }
        *a2 += v17;
        sub_8238B0(qword_4F18BE0, v18, v17);
      }
    }
  }
}
