// Function: sub_186A2D0
// Address: 0x186a2d0
//
__int64 __fastcall sub_186A2D0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 v3; // r13
  signed int v4; // r14d
  __int64 v5; // rbx
  __int64 v6; // r15
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  char v9; // cl
  unsigned __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r12
  int v16; // edi
  __int64 v17; // rax
  int v18; // edx
  char *v19; // rax
  char v20; // al
  __int64 result; // rax
  __int64 v22; // rax
  int v25; // [rsp+18h] [rbp-58h]
  int v26; // [rsp+1Ch] [rbp-54h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+28h] [rbp-48h]
  int i; // [rsp+30h] [rbp-40h]
  signed int v30; // [rsp+34h] [rbp-3Ch]

  v28 = *(_QWORD *)(a1 + 32);
  v27 = a1 + 24;
  v26 = 0;
  v25 = 0;
  for ( i = 0; v27 != v28; v28 = *(_QWORD *)(v28 + 8) )
  {
    if ( !v28 )
      BUG();
    v3 = *(_QWORD *)(v28 + 24);
    if ( v28 + 16 == v3 )
      continue;
    v30 = 0;
    v4 = 0;
    do
    {
      if ( !v3 )
        BUG();
      v5 = *(_QWORD *)(v3 + 24);
      v6 = v3 + 16;
      if ( v5 != v3 + 16 )
      {
        while ( 1 )
        {
          if ( !v5 )
            BUG();
          v7 = *(_BYTE *)(v5 - 8);
          if ( v7 == 27 )
          {
            if ( v4 < (int)(((*(_DWORD *)(v5 - 4) & 0xFFFFFFFu) >> 1) - 1) )
              v4 = ((*(_DWORD *)(v5 - 4) & 0xFFFFFFFu) >> 1) - 1;
            goto LABEL_10;
          }
          if ( v7 <= 0x17u )
            goto LABEL_10;
          v8 = v5 - 24;
          if ( v7 == 78 )
          {
            v9 = v8 | 4;
            v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v10 )
              goto LABEL_10;
            v22 = *(_QWORD *)(v5 - 48);
            if ( !*(_BYTE *)(v22 + 16) && (*(_BYTE *)(v22 + 33) & 0x20) != 0 )
              goto LABEL_10;
LABEL_17:
            v11 = (__int64 *)(v10 - 24);
            v12 = (__int64 *)(v10 - 72);
            if ( (v9 & 4) != 0 )
              v12 = v11;
            v13 = *v12;
            if ( *(_BYTE *)(*v12 + 16) || sub_15E4F60(*v12) )
              goto LABEL_10;
            if ( (unsigned __int8)sub_1560180(v13 + 112, 3) )
            {
              v14 = *(_QWORD *)(v13 + 80);
              v15 = v13 + 72;
              if ( v14 != v15 )
              {
                v16 = 0;
                do
                {
                  while ( 1 )
                  {
                    if ( !v14 )
                      BUG();
                    v17 = *(_QWORD *)(v14 + 24);
                    if ( v14 + 16 != v17 )
                      break;
                    v14 = *(_QWORD *)(v14 + 8);
                    if ( v15 == v14 )
                      goto LABEL_29;
                  }
                  v18 = 0;
                  do
                  {
                    v17 = *(_QWORD *)(v17 + 8);
                    ++v18;
                  }
                  while ( v14 + 16 != v17 );
                  v14 = *(_QWORD *)(v14 + 8);
                  v16 += v18;
                }
                while ( v15 != v14 );
LABEL_29:
                i += v16;
              }
            }
            v5 = *(_QWORD *)(v5 + 8);
            ++v30;
            if ( v6 == v5 )
              break;
          }
          else
          {
            if ( v7 == 29 )
            {
              v9 = v8 & 0xFB;
              v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v10 )
                goto LABEL_17;
            }
LABEL_10:
            v5 = *(_QWORD *)(v5 + 8);
            if ( v6 == v5 )
              break;
          }
        }
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v28 + 16 != v3 );
    if ( v4 <= v30 && v4 > 1 )
    {
      ++v25;
      if ( v26 >= v4 )
        v4 = v26;
      v26 = v4;
    }
  }
  *a2 = dword_4FABD20 - i;
  v19 = (char *)sub_16D40F0((__int64)qword_4FBB3F0);
  if ( v19 )
    v20 = *v19;
  else
    v20 = qword_4FBB3F0[2];
  if ( v20 || !byte_4FAB9A0 || dword_4FAB8C0 > v25 || (result = (unsigned int)dword_4FABA80, v26 < dword_4FAB7E0) )
    result = (unsigned int)dword_4FABB60;
  *a3 = result;
  return result;
}
