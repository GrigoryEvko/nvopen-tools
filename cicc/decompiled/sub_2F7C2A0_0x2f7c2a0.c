// Function: sub_2F7C2A0
// Address: 0x2f7c2a0
//
__int64 __fastcall sub_2F7C2A0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 i; // r15
  int v5; // eax
  unsigned __int8 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rsi
  _BYTE *v9; // r13
  __int64 v11; // r13
  size_t v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 j; // rdx
  unsigned __int8 v18; // [rsp+7h] [rbp-49h]
  _QWORD *v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]

  v2 = a2[6];
  if ( *(_BYTE *)(v2 + 66) || (v18 = *(_BYTE *)(v2 + 670)) != 0 )
  {
    v3 = (_QWORD *)a2[41];
    v19 = a2 + 40;
    if ( v3 == a2 + 40 )
    {
      return 0;
    }
    else
    {
      v18 = 0;
      v20 = *(_QWORD *)(*a2 + 40LL);
      do
      {
        for ( i = v3[7]; v3 + 6 != (_QWORD *)i; i = *(_QWORD *)(i + 8) )
        {
          v5 = *(_DWORD *)(i + 44);
          if ( (v5 & 4) != 0 || (v5 & 8) == 0 )
            v6 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) >> 7;
          else
            v6 = sub_2E88A90(i, 128, 1);
          if ( v6 )
          {
            v7 = *(_QWORD *)(i + 32);
            v8 = v7 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
            if ( v7 != v8 )
            {
              while ( 1 )
              {
                if ( *(_BYTE *)v7 == 10 )
                {
                  v9 = *(_BYTE **)(v7 + 24);
                  if ( *v9 || (unsigned __int8)sub_B2FC00(*(_BYTE **)(v7 + 24)) )
                    goto LABEL_17;
                  goto LABEL_29;
                }
                if ( *(_BYTE *)v7 == 9 )
                  break;
                v7 += 40;
                if ( v8 == v7 )
                  goto LABEL_17;
              }
              v11 = *(_QWORD *)(v7 + 24);
              v12 = 0;
              if ( v11 )
                v12 = strlen(*(const char **)(v7 + 24));
              v13 = sub_BA8CB0(v20, v11, v12);
              v9 = v13;
              if ( v13 && !(unsigned __int8)sub_B2FC00(v13) )
              {
LABEL_29:
                v14 = sub_2F7A490(*a1, (__int64)v9);
                if ( v15 )
                {
                  v16 = *(_QWORD *)(i + 32);
                  for ( j = v16 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF); j != v16; v16 += 40 )
                  {
                    if ( *(_BYTE *)v16 == 12 )
                      *(_QWORD *)(v16 + 24) = v14;
                  }
                  v18 = v6;
                }
              }
            }
          }
LABEL_17:
          if ( (*(_BYTE *)i & 4) == 0 )
          {
            while ( (*(_BYTE *)(i + 44) & 8) != 0 )
              i = *(_QWORD *)(i + 8);
          }
        }
        v3 = (_QWORD *)v3[1];
      }
      while ( v19 != v3 );
    }
  }
  return v18;
}
