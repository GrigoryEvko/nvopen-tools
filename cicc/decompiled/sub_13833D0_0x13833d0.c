// Function: sub_13833D0
// Address: 0x13833d0
//
char __fastcall sub_13833D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  char result; // al
  __int64 v10; // rcx
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // r9
  char *v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdi
  char *v19; // rsi
  char *v20; // rcx
  __int64 v21; // r9
  unsigned __int64 *v22; // rdi
  char *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int64 *v26; // rdx
  __int64 v27; // rdx
  int i; // edx
  int v29; // r10d
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h] BYREF
  char v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  char v35; // [rsp+28h] [rbp-38h]

  sub_1383340((__int64)&v32, a1, a2);
  sub_1383340((__int64)&v34, a1, a4);
  if ( !v33 || !v35 )
    return 1;
  v30 = v32;
  v31 = v34;
  if ( (unsigned __int8)sub_14C8180(v32) )
    return v31 != 0;
  if ( (unsigned __int8)sub_14C8180(v31) )
    return v30 != 0;
  if ( (unsigned __int8)sub_14C8210(v30) )
    return sub_14C8210(v31);
  result = sub_14C8210(v31);
  if ( result )
    return sub_14C8210(v30);
  v10 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v11 + 32LL * (((_DWORD)v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v14 = *v13;
    if ( a2 != *v13 )
    {
      for ( i = 1; ; i = v29 )
      {
        if ( v14 == -8 )
          return result;
        v29 = i + 1;
        v12 = (v10 - 1) & (i + v12);
        v13 = (__int64 *)(v11 + 32LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          break;
      }
    }
    if ( v13 != (__int64 *)(v11 + 32 * v10) )
    {
      v15 = (char *)v13[1];
      v16 = v13[2] - (_QWORD)v15;
      v17 = v16 >> 4;
      if ( v16 > 0 )
      {
        do
        {
          v18 = 16 * (v17 >> 1);
          v19 = &v15[v18];
          if ( *(_QWORD *)&v15[v18] < a4 )
          {
            v15 = v19 + 16;
            v17 = v17 - (v17 >> 1) - 1;
          }
          else
          {
            if ( *(_QWORD *)&v15[v18] <= a4 )
            {
              v20 = v15;
              v21 = v18 >> 4;
              if ( v18 )
              {
                do
                {
                  v22 = (unsigned __int64 *)&v20[16 * (v21 >> 1)];
                  if ( *v22 < a4 )
                  {
                    v20 = (char *)(v22 + 2);
                    v21 = v21 - (v21 >> 1) - 1;
                  }
                  else
                  {
                    v21 >>= 1;
                  }
                }
                while ( v21 > 0 );
              }
              v23 = v19 + 16;
              v24 = &v15[16 * v17] - v23;
              v25 = v24 >> 4;
              if ( v24 > 0 )
              {
                do
                {
                  v26 = (unsigned __int64 *)&v23[16 * (v25 >> 1)];
                  if ( *v26 <= a4 )
                  {
                    v23 = (char *)(v26 + 2);
                    v25 = v25 - (v25 >> 1) - 1;
                  }
                  else
                  {
                    v25 >>= 1;
                  }
                }
                while ( v25 > 0 );
              }
              if ( v20 != v23 )
              {
                if ( a3 != -1 && a5 != -1 )
                {
                  while ( 1 )
                  {
                    v27 = *((_QWORD *)v20 + 1);
                    if ( v27 == 0x7FFFFFFFFFFFFFFFLL || a3 < 0 || a5 < 0 || v27 + a3 > 0 && v27 < a5 )
                      break;
                    v20 += 16;
                    if ( v20 == v23 )
                      return result;
                  }
                }
                return 1;
              }
              return result;
            }
            v17 >>= 1;
          }
        }
        while ( v17 > 0 );
      }
    }
  }
  return result;
}
