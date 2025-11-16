// Function: sub_20D6E00
// Address: 0x20d6e00
//
__int64 __fastcall sub_20D6E00(__int64 a1, _QWORD *a2)
{
  __int64 v4; // r13
  __int64 i; // rsi
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 *v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 result; // rax
  int v12; // ecx
  __int64 v13; // rsi
  unsigned int v14; // edx
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // rcx
  int v18; // r8d
  unsigned int v19; // edx
  _QWORD *v20; // r13
  _QWORD *v21; // rsi
  __int64 *v22; // r14
  char *v23; // rdx
  char *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rcx
  char *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  _QWORD *v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r8d

  v4 = a2[7];
  for ( i = a2[12]; i != a2[11]; i = a2[12] )
    sub_1DD9130((__int64)a2, (__int64 *)(i - 8), 0);
  v6 = *(_QWORD **)(a1 + 32);
  if ( *(_QWORD **)(a1 + 40) == v6 )
  {
    v30 = &v6[*(unsigned int *)(a1 + 52)];
    if ( v6 == v30 )
    {
LABEL_55:
      v6 = v30;
    }
    else
    {
      while ( (_QWORD *)*v6 != a2 )
      {
        if ( v30 == ++v6 )
          goto LABEL_55;
      }
    }
  }
  else
  {
    v6 = sub_16CC9F0(a1 + 24, (__int64)a2);
    if ( (_QWORD *)*v6 == a2 )
    {
      v28 = *(_QWORD *)(a1 + 40);
      if ( v28 == *(_QWORD *)(a1 + 32) )
        v29 = *(unsigned int *)(a1 + 52);
      else
        v29 = *(unsigned int *)(a1 + 48);
      v30 = (_QWORD *)(v28 + 8 * v29);
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 40);
      if ( v7 != *(_QWORD *)(a1 + 32) )
        goto LABEL_6;
      v6 = (_QWORD *)(v7 + 8LL * *(unsigned int *)(a1 + 52));
      v30 = v6;
    }
  }
  if ( v30 != v6 )
  {
    *v6 = -2;
    ++*(_DWORD *)(a1 + 56);
  }
LABEL_6:
  v8 = v4 + 320;
  sub_1DD5B80(v8, (__int64)a2);
  v9 = (unsigned __int64 *)a2[1];
  v10 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v9 = v10 | *v9 & 7;
  *(_QWORD *)(v10 + 8) = v9;
  *a2 &= 7uLL;
  a2[1] = 0;
  sub_1E0A230(v8, a2);
  result = *(unsigned int *)(a1 + 104);
  if ( (_DWORD)result )
  {
    v12 = result - 1;
    v13 = *(_QWORD *)(a1 + 88);
    v14 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v13 + 16LL * v14;
    v15 = *(_QWORD *)result;
    if ( *(_QWORD **)result == a2 )
    {
LABEL_8:
      *(_QWORD *)result = -16;
      --*(_DWORD *)(a1 + 96);
      ++*(_DWORD *)(a1 + 100);
    }
    else
    {
      result = 1;
      while ( v15 != -8 )
      {
        v34 = result + 1;
        v14 = v12 & (result + v14);
        result = v13 + 16LL * v14;
        v15 = *(_QWORD *)result;
        if ( *(_QWORD **)result == a2 )
          goto LABEL_8;
        result = v34;
      }
    }
  }
  v16 = *(_QWORD *)(a1 + 176);
  if ( v16 )
  {
    result = *(unsigned int *)(v16 + 256);
    if ( (_DWORD)result )
    {
      v17 = *(_QWORD *)(v16 + 240);
      v18 = 1;
      v19 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (_QWORD *)(v17 + 16LL * v19);
      v21 = (_QWORD *)*v20;
      if ( (_QWORD *)*v20 == a2 )
      {
LABEL_12:
        result = v17 + 16 * result;
        if ( v20 != (_QWORD *)result )
        {
          v22 = (__int64 *)v20[1];
          if ( v22 )
          {
            while ( 1 )
            {
              v23 = (char *)v22[5];
              v24 = (char *)v22[4];
              v25 = (v23 - v24) >> 5;
              v26 = (v23 - v24) >> 3;
              if ( v25 <= 0 )
                break;
              v27 = &v24[32 * v25];
              while ( *(_QWORD **)v24 != a2 )
              {
                if ( *((_QWORD **)v24 + 1) == a2 )
                {
                  v24 += 8;
                  break;
                }
                if ( *((_QWORD **)v24 + 2) == a2 )
                {
                  v24 += 16;
                  break;
                }
                if ( *((_QWORD **)v24 + 3) == a2 )
                {
                  v24 += 24;
                  break;
                }
                v24 += 32;
                if ( v27 == v24 )
                {
                  v26 = (v23 - v24) >> 3;
                  goto LABEL_45;
                }
              }
LABEL_21:
              if ( v24 + 8 != v23 )
              {
                memmove(v24, v24 + 8, v23 - (v24 + 8));
                v23 = (char *)v22[5];
              }
              result = v22[8];
              v22[5] = (__int64)(v23 - 8);
              if ( v22[9] == result )
              {
                v31 = result + 8LL * *((unsigned int *)v22 + 21);
                if ( result == v31 )
                {
LABEL_43:
                  result = v31;
                }
                else
                {
                  while ( *(_QWORD **)result != a2 )
                  {
                    result += 8;
                    if ( v31 == result )
                      goto LABEL_43;
                  }
                }
                goto LABEL_38;
              }
              result = (__int64)sub_16CC9F0((__int64)(v22 + 7), (__int64)a2);
              if ( *(_QWORD **)result == a2 )
              {
                v32 = v22[9];
                if ( v32 == v22[8] )
                  v33 = *((unsigned int *)v22 + 21);
                else
                  v33 = *((unsigned int *)v22 + 20);
                v31 = v32 + 8 * v33;
                goto LABEL_38;
              }
              result = v22[9];
              if ( result == v22[8] )
              {
                result += 8LL * *((unsigned int *)v22 + 21);
                v31 = result;
LABEL_38:
                if ( v31 != result )
                {
                  *(_QWORD *)result = -2;
                  ++*((_DWORD *)v22 + 22);
                }
              }
              v22 = (__int64 *)*v22;
              if ( !v22 )
                goto LABEL_27;
            }
LABEL_45:
            if ( v26 != 2 )
            {
              if ( v26 != 3 )
              {
                if ( v26 != 1 )
                {
                  v24 = (char *)v22[5];
                  goto LABEL_21;
                }
LABEL_61:
                if ( *(_QWORD **)v24 != a2 )
                  v24 = (char *)v22[5];
                goto LABEL_21;
              }
              if ( *(_QWORD **)v24 == a2 )
                goto LABEL_21;
              v24 += 8;
            }
            if ( *(_QWORD **)v24 == a2 )
              goto LABEL_21;
            v24 += 8;
            goto LABEL_61;
          }
LABEL_27:
          *v20 = -16;
          --*(_DWORD *)(v16 + 248);
          ++*(_DWORD *)(v16 + 252);
        }
      }
      else
      {
        while ( v21 != (_QWORD *)-8LL )
        {
          v19 = (result - 1) & (v18 + v19);
          v20 = (_QWORD *)(v17 + 16LL * v19);
          v21 = (_QWORD *)*v20;
          if ( (_QWORD *)*v20 == a2 )
            goto LABEL_12;
          ++v18;
        }
      }
    }
  }
  return result;
}
