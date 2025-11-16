// Function: sub_B97D20
// Address: 0xb97d20
//
__int64 __fastcall sub_B97D20(__int64 *a1, int a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d
  __int64 v5; // rdx
  char *v6; // rbx
  char *v7; // r14
  __int64 v8; // r12
  __int64 v9; // rax
  char *v10; // rax
  char *i; // r12
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  char *v16; // r12
  __int64 v17; // r15
  __int64 *v18; // r14
  __int64 *v19; // r12
  unsigned __int8 *v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rbx
  __int64 v25; // rsi
  __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]

  v2 = *((_DWORD *)a1 + 2);
  v3 = 0;
  if ( v2 )
  {
    v5 = *a1;
    v27 = v2;
    v6 = (char *)*a1;
    if ( v2 == 1 )
    {
      v7 = (char *)(v5 + 16);
      if ( *(_DWORD *)v5 == a2 )
      {
        *((_DWORD *)a1 + 2) = 0;
        v25 = *(_QWORD *)(v5 + 8);
        if ( v25 )
          sub_B91220(v5 + 8, v25);
        return 1;
      }
      goto LABEL_4;
    }
    v7 = (char *)(v5 + 16LL * v2);
    v9 = (16LL * v2) >> 6;
    if ( v9 )
    {
      v10 = (char *)(v5 + (v9 << 6));
      while ( a2 != *(_DWORD *)v6 )
      {
        if ( a2 == *((_DWORD *)v6 + 4) )
        {
          v6 += 16;
          goto LABEL_13;
        }
        if ( a2 == *((_DWORD *)v6 + 8) )
        {
          v6 += 32;
          goto LABEL_13;
        }
        if ( a2 == *((_DWORD *)v6 + 12) )
        {
          v6 += 48;
          goto LABEL_13;
        }
        v6 += 64;
        if ( v10 == v6 )
          goto LABEL_49;
      }
      goto LABEL_13;
    }
LABEL_49:
    if ( v7 - v6 != 32 )
    {
      if ( v7 - v6 != 48 )
      {
        if ( (v7 - v6) >> 4 != 1 )
          goto LABEL_5;
LABEL_4:
        if ( a2 != *(_DWORD *)v6 )
        {
LABEL_5:
          v8 = v27;
          v6 = v7;
LABEL_26:
          v16 = (char *)(v5 + 16 * v8);
          v26 = v16 - v7;
          v17 = (v16 - v7) >> 4;
          if ( v16 - v7 > 0 )
          {
            v18 = (__int64 *)(v7 + 8);
            v19 = (__int64 *)(v6 + 8);
            do
            {
              *((_DWORD *)v19 - 2) = *((_DWORD *)v18 - 2);
              if ( v19 != v18 )
              {
                if ( *v19 )
                  sub_B91220((__int64)v19, *v19);
                v20 = (unsigned __int8 *)*v18;
                *v19 = *v18;
                if ( v20 )
                {
                  sub_B976B0((__int64)v18, v20, (__int64)v19);
                  *v18 = 0;
                }
              }
              v18 += 2;
              v19 += 2;
              --v17;
            }
            while ( v17 );
            v21 = v26;
            v5 = *a1;
            if ( v26 <= 0 )
              v21 = 16;
            v16 = (char *)(v5 + 16LL * *((unsigned int *)a1 + 2));
            v6 += v21;
          }
          if ( v6 != v16 )
          {
            do
            {
              v22 = *((_QWORD *)v16 - 1);
              v16 -= 16;
              if ( v22 )
                sub_B91220((__int64)(v16 + 8), v22);
            }
            while ( v16 != v6 );
            v5 = *a1;
          }
          v23 = (__int64)&v6[-v5] >> 4;
          *((_DWORD *)a1 + 2) = v23;
          LOBYTE(v3) = (unsigned int)v23 != v27;
          return v3;
        }
LABEL_13:
        if ( v6 == v7 || v7 == v6 + 16 )
        {
          v8 = v27;
        }
        else
        {
          for ( i = v6 + 24; ; i += 16 )
          {
            v12 = *((_DWORD *)i - 2);
            if ( a2 != v12 )
            {
              v13 = (__int64)(v6 + 8);
              *(_DWORD *)v6 = v12;
              if ( v6 + 8 != i )
              {
                v14 = *((_QWORD *)v6 + 1);
                if ( v14 )
                {
                  sub_B91220((__int64)(v6 + 8), v14);
                  v13 = (__int64)(v6 + 8);
                }
                v15 = *(unsigned __int8 **)i;
                *((_QWORD *)v6 + 1) = *(_QWORD *)i;
                if ( v15 )
                {
                  sub_B976B0((__int64)i, v15, v13);
                  *(_QWORD *)i = 0;
                }
              }
              v6 += 16;
            }
            if ( v7 == i + 8 )
              break;
          }
          v8 = *((unsigned int *)a1 + 2);
          v5 = *a1;
        }
        goto LABEL_26;
      }
      if ( a2 == *(_DWORD *)v6 )
        goto LABEL_13;
      v6 += 16;
    }
    if ( a2 == *(_DWORD *)v6 )
      goto LABEL_13;
    v6 += 16;
    goto LABEL_4;
  }
  return 0;
}
