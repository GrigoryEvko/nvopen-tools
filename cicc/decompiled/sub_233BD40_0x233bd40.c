// Function: sub_233BD40
// Address: 0x233bd40
//
char __fastcall sub_233BD40(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  char result; // al
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // r9
  __int64 v12; // r9
  __int64 v13; // r9
  const void *v14; // rdi
  _QWORD *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-48h]
  char v26; // [rsp+8h] [rbp-48h]
  char v27; // [rsp+8h] [rbp-48h]
  char v28; // [rsp+8h] [rbp-48h]
  char v29; // [rsp+8h] [rbp-48h]
  char v30; // [rsp+8h] [rbp-48h]
  char v31; // [rsp+8h] [rbp-48h]
  _QWORD v32[8]; // [rsp+10h] [rbp-40h] BYREF

  result = sub_9691B0(a3, a4, "globals-aa", 10);
  if ( result )
  {
    v19 = *(unsigned int *)(a2 + 8);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v26 = result;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v19 + 1, 8u, v19 + 1, v9);
      v19 = *(unsigned int *)(a2 + 8);
      result = v26;
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v19) = sub_2396EC0;
    ++*(_DWORD *)(a2 + 8);
  }
  else
  {
    result = sub_9691B0(a3, a4, "basic-aa", 8);
    if ( result )
    {
      v21 = *(unsigned int *)(a2 + 8);
      if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v27 = result;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v21 + 1, 8u, v21 + 1, v10);
        v21 = *(unsigned int *)(a2 + 8);
        result = v27;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v21) = sub_2361CE0;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      result = sub_9691B0(a3, a4, "objc-arc-aa", 11);
      if ( result )
      {
        v22 = *(unsigned int *)(a2 + 8);
        if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v28 = result;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v22 + 1, 8u, v22 + 1, v11);
          v22 = *(unsigned int *)(a2 + 8);
          result = v28;
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v22) = sub_2361F60;
        ++*(_DWORD *)(a2 + 8);
      }
      else
      {
        result = sub_9691B0(a3, a4, "scev-aa", 7);
        if ( result )
        {
          v20 = *(unsigned int *)(a2 + 8);
          if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v29 = result;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v20 + 1, 8u, v20 + 1, v12);
            v20 = *(unsigned int *)(a2 + 8);
            result = v29;
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v20) = sub_2362040;
          ++*(_DWORD *)(a2 + 8);
        }
        else
        {
          result = sub_9691B0(a3, a4, "scoped-noalias-aa", 17);
          if ( result )
          {
            v23 = *(unsigned int *)(a2 + 8);
            if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
            {
              v30 = result;
              sub_C8D5F0(a2, (const void *)(a2 + 16), v23 + 1, 8u, v23 + 1, v13);
              v23 = *(unsigned int *)(a2 + 8);
              result = v30;
            }
            *(_QWORD *)(*(_QWORD *)a2 + 8 * v23) = sub_2362120;
            ++*(_DWORD *)(a2 + 8);
          }
          else
          {
            v14 = a3;
            v15 = (_QWORD *)a4;
            result = sub_9691B0(a3, a4, "tbaa", 4);
            if ( result )
            {
              v24 = *(unsigned int *)(a2 + 8);
              if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
              {
                v31 = result;
                sub_C8D5F0(a2, (const void *)(a2 + 16), v24 + 1, 8u, v24 + 1, v17);
                v24 = *(unsigned int *)(a2 + 8);
                result = v31;
              }
              *(_QWORD *)(*(_QWORD *)a2 + 8 * v24) = sub_2362200;
              ++*(_DWORD *)(a2 + 8);
            }
            else
            {
              v18 = *(_QWORD *)(a1 + 1968);
              v25 = v18 + 32LL * *(unsigned int *)(a1 + 1976);
              if ( v25 == v18 )
              {
                return 0;
              }
              else
              {
                while ( 1 )
                {
                  v32[0] = a3;
                  v32[1] = a4;
                  if ( !*(_QWORD *)(v18 + 16) )
                    sub_4263D6(v14, v15, v16);
                  v15 = v32;
                  v14 = (const void *)v18;
                  result = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64))(v18 + 24))(v18, v32, a2);
                  if ( result )
                    break;
                  v18 += 32;
                  if ( v18 == v25 )
                    return 0;
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
