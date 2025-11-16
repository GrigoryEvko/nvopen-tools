// Function: sub_7A87A0
// Address: 0x7a87a0
//
__int64 __fastcall sub_7A87A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 **v5; // rax
  __int64 *v6; // rsi
  __int64 *v7; // rcx
  __int64 v8; // rsi
  unsigned __int64 v9; // r15
  _DWORD *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r8
  int v13; // r12d
  __int64 v14; // rdi
  __int64 result; // rax
  unsigned __int64 v16; // r13
  char v17; // al
  unsigned __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // rax
  unsigned int v24; // eax
  unsigned int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rcx
  __int64 **v28; // r15
  _QWORD *v29; // r13
  __int64 i; // rdi
  __int64 j; // rdi
  __int64 v32; // [rsp+8h] [rbp-48h]
  unsigned int v33[14]; // [rsp+18h] [rbp-38h] BYREF

  if ( dword_4D0425C )
  {
    if ( (*(_BYTE *)(a2 + 96) & 2) != 0 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 56) + 168LL) + 16LL);
      if ( v4 )
      {
        if ( a2 != v4 )
        {
          do
          {
            if ( (*(_BYTE *)(v4 + 96) & 0x41) == 0x41 )
            {
              v5 = *(__int64 ***)(*(_QWORD *)(v4 + 40) + 168LL);
              while ( 1 )
              {
                v5 = (__int64 **)*v5;
                if ( !v5 )
                  break;
                if ( ((_BYTE)v5[12] & 2) != 0 )
                {
                  v6 = v5[5];
                  v7 = *(__int64 **)(a2 + 40);
                  if ( v6 == v7 || v7 && v6 && dword_4F07588 && (v8 = v6[4], v7[4] == v8) && v8 )
                  {
                    v9 = (unsigned __int64)v5[13];
                    if ( !v9 )
                      goto LABEL_21;
                    v10 = (_DWORD *)a2;
                    if ( (unsigned int)sub_7A8650((__int64 *)a1, a2, (unsigned __int64)v5[13]) )
                      goto LABEL_36;
                    goto LABEL_20;
                  }
                }
              }
            }
            v4 = *(_QWORD *)(v4 + 16);
          }
          while ( a2 != v4 && v4 );
        }
      }
    }
  }
LABEL_21:
  v10 = (_DWORD *)a2;
  v13 = sub_7A8650((__int64 *)a1, a2, 0);
  if ( v13 )
    goto LABEL_35;
  if ( !dword_4D0425C )
    goto LABEL_23;
  v28 = **(__int64 ****)(*(_QWORD *)a1 + 168LL);
  if ( !v28 )
    goto LABEL_23;
  do
  {
    if ( ((_BYTE)v28[12] & 0x41) == 0x41 && !v28[13] )
    {
      v29 = **(_QWORD ***)(*(_QWORD *)(a2 + 40) + 168LL);
      if ( v29 )
      {
        while ( 1 )
        {
          if ( !**(_QWORD **)(v29[5] + 168LL) )
          {
            for ( i = (__int64)v28[5]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v10 = v29;
            if ( (unsigned int)sub_7A6F40(i, (__int64)v29, 1u, (*(_BYTE *)(a2 + 96) & 2) != 0, v12) )
              break;
          }
          v29 = (_QWORD *)*v29;
          if ( !v29 )
            goto LABEL_63;
        }
        v13 = 1;
      }
      else
      {
        for ( j = (__int64)v28[5]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v10 = (_DWORD *)a2;
        if ( (unsigned int)sub_7A6F40(j, a2, 1u, (*(_BYTE *)(a2 + 96) & 2) != 0, v12) )
          goto LABEL_35;
      }
    }
LABEL_63:
    v28 = (__int64 **)*v28;
  }
  while ( v28 );
  if ( v13 )
  {
LABEL_35:
    v9 = 0;
LABEL_36:
    v16 = *(_QWORD *)(a1 + 8);
    v17 = *(_BYTE *)(a2 + 96);
    if ( dword_4D0425C )
    {
      if ( (v17 & 2) != 0 && (unsigned __int64)(unk_4D04250 - 30300LL) <= 0x63 )
      {
        v19 = *(_QWORD *)(a1 + 72);
        if ( v19 )
        {
          if ( (*(_BYTE *)(v19 + 96) & 0x42) == 0x42 )
          {
            v20 = *(_QWORD *)(v19 + 40);
            v21 = *(_QWORD *)(v19 + 104);
            if ( v16 <= v21 + *(_QWORD *)(v20 + 128) )
            {
              if ( !*(_QWORD *)v20 )
                goto LABEL_55;
              v22 = *(_QWORD *)(v19 + 40);
              if ( *(_BYTE *)(v20 + 140) == 12 )
              {
                do
                  v22 = *(_QWORD *)(v22 + 160);
                while ( *(_BYTE *)(v22 + 140) == 12 );
              }
              if ( *(char *)(*(_QWORD *)(*(_QWORD *)v22 + 96LL) + 178LL) >= 0 )
              {
LABEL_55:
                *(_QWORD *)v33 = v21;
                v32 = v20;
                v23 = (unsigned __int8 *)sub_7A6650(v20, v33);
                if ( v23 )
                {
                  if ( (v23[144] & 4) != 0 )
                  {
                    v24 = v23[137] + v23[136];
                    v26 = v24 % dword_4F06BA0;
                    v25 = v24 / dword_4F06BA0;
                    if ( v26 )
                    {
                      v27 = *(unsigned int *)(v32 + 136);
                      if ( v16 < *(_QWORD *)v33 + v27 + (unsigned __int64)v25 + 1 )
                        --v16;
                      else
                        v16 -= (unsigned int)(v27 - 1);
                    }
                  }
                }
                v17 = *(_BYTE *)(a2 + 96);
              }
            }
          }
        }
      }
    }
    v9 += v16;
    if ( (v17 & 1) != 0 )
      *(_QWORD *)(a1 + 8) = v9;
    v18 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL) + 40LL);
    while ( 1 )
    {
      v10 = (_DWORD *)a2;
      if ( !(unsigned int)sub_7A8650((__int64 *)a1, a2, v9) )
        break;
      if ( unk_4F06AC0 < v18 || unk_4F06AC0 - v18 < v9 )
      {
        if ( !*(_BYTE *)(a1 + 28) )
        {
          v10 = dword_4F07508;
          sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
          *(_BYTE *)(a1 + 28) = 1;
          *(_QWORD *)(a2 + 104) = v9;
          goto LABEL_24;
        }
      }
      else
      {
        v9 += v18;
      }
    }
LABEL_20:
    *(_QWORD *)(a2 + 104) = v9;
  }
  else
  {
LABEL_23:
    *(_QWORD *)(a2 + 104) = 0;
  }
LABEL_24:
  v14 = *(_QWORD *)(a2 + 40);
  *(_BYTE *)(a2 + 96) |= 0x20u;
  if ( *(char *)(v14 + 142) >= 0 && *(_BYTE *)(v14 + 140) == 12 )
    result = sub_8D4AB0(v14, v10, v11);
  else
    result = *(unsigned int *)(v14 + 136);
  v33[0] = result;
  if ( unk_4F06A74 )
  {
    sub_7A65D0(v33, *(_QWORD *)(a2 + 56));
    result = v33[0];
  }
  if ( *(_DWORD *)(a1 + 24) < (unsigned int)result && (!dword_4D0425C || unk_4D04250 > 0x9D6Bu) )
    *(_DWORD *)(a1 + 24) = result;
  return result;
}
