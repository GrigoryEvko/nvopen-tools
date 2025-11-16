// Function: sub_34BEF40
// Address: 0x34bef40
//
__int64 __fastcall sub_34BEF40(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 i; // r14
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  __int64 j; // r15
  __int64 v12; // r14
  unsigned __int64 *v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rsi
  int v17; // ecx
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 *v23; // r13
  __int64 v24; // rsi
  __int64 *k; // r14
  char *v26; // rdx
  char *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rcx
  char *v30; // rax
  bool v31; // zf
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 *v35; // rax
  unsigned int v36; // r8d
  int v37; // r8d

  v4 = *(_DWORD *)(a2 + 120);
  for ( i = *(_QWORD *)(a2 + 32); v4; v4 = *(_DWORD *)(a2 + 120) )
    sub_2E33590(a2, (__int64 *)(*(_QWORD *)(a2 + 112) + 8LL * v4 - 8), 0);
  if ( *(_BYTE *)(a1 + 52) )
  {
    v6 = *(_QWORD **)(a1 + 32);
    v7 = &v6[*(unsigned int *)(a1 + 44)];
    v8 = v6;
    if ( v6 != v7 )
    {
      while ( *v8 != a2 )
      {
        if ( v7 == ++v8 )
          goto LABEL_9;
      }
      v9 = (unsigned int)(*(_DWORD *)(a1 + 44) - 1);
      *(_DWORD *)(a1 + 44) = v9;
      *v8 = v6[v9];
      ++*(_QWORD *)(a1 + 24);
    }
  }
  else
  {
    v35 = sub_C8CA60(a1 + 24, a2);
    if ( v35 )
    {
      *v35 = -2;
      ++*(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 24);
    }
  }
LABEL_9:
  v10 = *(_QWORD *)(a2 + 56);
  for ( j = a2 + 48; j != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    while ( 1 )
    {
      if ( sub_2E88F60(v10) )
        sub_2E79700(i, v10);
      if ( !v10 )
        BUG();
      if ( (*(_BYTE *)v10 & 4) == 0 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( j == v10 )
        goto LABEL_17;
    }
    while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
      v10 = *(_QWORD *)(v10 + 8);
  }
LABEL_17:
  v12 = i + 320;
  sub_2E31020(v12, a2);
  v13 = *(unsigned __int64 **)(a2 + 8);
  v14 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v13 = v14 | *v13 & 7;
  *(_QWORD *)(v14 + 8) = v13;
  *(_QWORD *)a2 &= 7uLL;
  *(_QWORD *)(a2 + 8) = 0;
  sub_2E79D60(v12, (_QWORD *)a2);
  result = *(unsigned int *)(a1 + 96);
  v16 = *(_QWORD *)(a1 + 80);
  if ( (_DWORD)result )
  {
    v17 = result - 1;
    v18 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v16 + 16LL * v18;
    v19 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
    {
LABEL_19:
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(a1 + 88);
      ++*(_DWORD *)(a1 + 92);
    }
    else
    {
      result = 1;
      while ( v19 != -4096 )
      {
        v36 = result + 1;
        v18 = v17 & (result + v18);
        result = v16 + 16LL * v18;
        v19 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
          goto LABEL_19;
        result = v36;
      }
    }
  }
  v20 = *(_QWORD *)(a1 + 160);
  if ( v20 )
  {
    result = *(unsigned int *)(v20 + 24);
    v21 = *(_QWORD *)(v20 + 8);
    if ( (_DWORD)result )
    {
      v22 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( *v23 == a2 )
      {
LABEL_23:
        result = v21 + 16 * result;
        if ( v23 != (__int64 *)result )
        {
          for ( k = (__int64 *)v23[1]; k; k = (__int64 *)*k )
          {
            v26 = (char *)k[5];
            v27 = (char *)k[4];
            v28 = (v26 - v27) >> 5;
            v29 = (v26 - v27) >> 3;
            if ( v28 > 0 )
            {
              v30 = &v27[32 * v28];
              while ( *(_QWORD *)v27 != a2 )
              {
                if ( *((_QWORD *)v27 + 1) == a2 )
                {
                  v27 += 8;
                  goto LABEL_32;
                }
                if ( *((_QWORD *)v27 + 2) == a2 )
                {
                  v27 += 16;
                  goto LABEL_32;
                }
                if ( *((_QWORD *)v27 + 3) == a2 )
                {
                  v27 += 24;
                  goto LABEL_32;
                }
                v27 += 32;
                if ( v30 == v27 )
                {
                  v29 = (v26 - v27) >> 3;
                  goto LABEL_49;
                }
              }
              goto LABEL_32;
            }
LABEL_49:
            if ( v29 != 2 )
            {
              if ( v29 != 3 )
              {
                if ( v29 == 1 )
                  goto LABEL_60;
                v27 = (char *)k[5];
                goto LABEL_32;
              }
              if ( *(_QWORD *)v27 == a2 )
                goto LABEL_32;
              v27 += 8;
            }
            if ( *(_QWORD *)v27 != a2 )
            {
              v27 += 8;
LABEL_60:
              if ( *(_QWORD *)v27 != a2 )
                v27 = (char *)k[5];
            }
LABEL_32:
            if ( v27 + 8 != v26 )
            {
              memmove(v27, v27 + 8, v26 - (v27 + 8));
              v26 = (char *)k[5];
            }
            v31 = *((_BYTE *)k + 84) == 0;
            k[5] = (__int64)(v26 - 8);
            if ( v31 )
            {
              result = (__int64)sub_C8CA60((__int64)(k + 7), a2);
              if ( result )
              {
                *(_QWORD *)result = -2;
                ++*((_DWORD *)k + 20);
                ++k[7];
              }
            }
            else
            {
              v32 = k[8];
              v33 = v32 + 8LL * *((unsigned int *)k + 19);
              result = v32;
              if ( v32 != v33 )
              {
                while ( *(_QWORD *)result != a2 )
                {
                  result += 8;
                  if ( v33 == result )
                    goto LABEL_40;
                }
                v34 = (unsigned int)(*((_DWORD *)k + 19) - 1);
                *((_DWORD *)k + 19) = v34;
                *(_QWORD *)result = *(_QWORD *)(v32 + 8 * v34);
                ++k[7];
              }
            }
LABEL_40:
            ;
          }
          *v23 = -8192;
          --*(_DWORD *)(v20 + 16);
          ++*(_DWORD *)(v20 + 20);
        }
      }
      else
      {
        v37 = 1;
        while ( v24 != -4096 )
        {
          v22 = (result - 1) & (v37 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( *v23 == a2 )
            goto LABEL_23;
          ++v37;
        }
      }
    }
  }
  return result;
}
