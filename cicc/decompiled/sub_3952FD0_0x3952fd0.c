// Function: sub_3952FD0
// Address: 0x3952fd0
//
__int64 __fastcall sub_3952FD0(__int64 a1)
{
  size_t v1; // rdx
  __int64 v3; // r14
  char *v4; // r13
  __int64 v5; // rsi
  __int64 result; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 *v13; // rsi
  unsigned int v14; // edi
  _QWORD *v15; // rcx

  v1 = 0;
  v3 = *(_QWORD *)a1;
  v4 = off_4CD4950[0];
  if ( off_4CD4950[0] )
    v1 = strlen(off_4CD4950[0]);
  v5 = (__int64)v4;
  result = sub_1626CE0(v3, v4, v1);
  if ( result )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(result - 8LL * *(unsigned int *)(result + 8)) + 136LL);
    v8 = *(_QWORD *)(v7 + 24);
    if ( *(_DWORD *)(v7 + 32) > 0x40u )
      v8 = *(_QWORD *)v8;
    v9 = *(_QWORD *)a1;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 18LL) & 1) != 0 )
    {
      sub_15E08E0(*(_QWORD *)a1, v5);
      v10 = *(_QWORD *)(v9 + 88);
      result = 5LL * *(_QWORD *)(v9 + 96);
      v11 = v10 + 40LL * *(_QWORD *)(v9 + 96);
      if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
      {
        result = sub_15E08E0(v9, v5);
        v10 = *(_QWORD *)(v9 + 88);
      }
    }
    else
    {
      v10 = *(_QWORD *)(v9 + 88);
      result = 5LL * *(_QWORD *)(v9 + 96);
      v11 = v10 + 40LL * *(_QWORD *)(v9 + 96);
    }
    if ( v11 != v10 )
    {
      while ( 1 )
      {
        result = (unsigned int)(1 << *(_DWORD *)(v10 + 32));
        if ( (v8 & result) == 0 )
          goto LABEL_10;
        result = *(_QWORD *)(a1 + 184);
        if ( *(_QWORD *)(a1 + 192) == result )
        {
          v13 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 204));
          v14 = *(_DWORD *)(a1 + 204);
          if ( (__int64 *)result == v13 )
          {
LABEL_25:
            if ( v14 >= *(_DWORD *)(a1 + 200) )
              goto LABEL_13;
            *(_DWORD *)(a1 + 204) = v14 + 1;
            *v13 = v10;
            ++*(_QWORD *)(a1 + 176);
          }
          else
          {
            v15 = 0;
            while ( *(_QWORD *)result != v10 )
            {
              if ( *(_QWORD *)result == -2 )
                v15 = (_QWORD *)result;
              result += 8;
              if ( v13 == (__int64 *)result )
              {
                if ( !v15 )
                  goto LABEL_25;
                *v15 = v10;
                --*(_DWORD *)(a1 + 208);
                ++*(_QWORD *)(a1 + 176);
                break;
              }
            }
          }
LABEL_10:
          v10 += 40;
          if ( v10 == v11 )
            return result;
        }
        else
        {
LABEL_13:
          v12 = v10;
          v10 += 40;
          result = (__int64)sub_16CCBA0(a1 + 176, v12);
          if ( v10 == v11 )
            return result;
        }
      }
    }
  }
  return result;
}
