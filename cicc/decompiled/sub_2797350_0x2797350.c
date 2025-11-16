// Function: sub_2797350
// Address: 0x2797350
//
__int64 __fastcall sub_2797350(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r11d
  unsigned int i; // eax
  __int64 v12; // rsi
  unsigned int v13; // eax
  unsigned int v14; // r15d
  unsigned int v16; // eax
  unsigned int v17; // esi
  int v18; // r10d
  __int64 v19; // rdx
  unsigned int *v20; // r9
  unsigned int j; // eax
  unsigned int *v22; // rcx
  unsigned int v23; // edi
  unsigned int v24; // eax
  int v25; // eax
  int v26; // edx
  unsigned int *v27; // [rsp+8h] [rbp-58h] BYREF
  unsigned int v28; // [rsp+10h] [rbp-50h] BYREF
  __int64 v29; // [rsp+18h] [rbp-48h]
  unsigned int v30; // [rsp+20h] [rbp-40h]

  v8 = *(unsigned int *)(a1 + 176);
  v9 = *(_QWORD *)(a1 + 160);
  if ( !(_DWORD)v8 )
    goto LABEL_11;
  v10 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned __int64)(37 * a4) << 32) | ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v8 - 1) & v13 )
  {
    v12 = v9 + 24LL * i;
    if ( a4 == *(_DWORD *)v12 && a2 == *(_QWORD *)(v12 + 8) )
      break;
    if ( *(_DWORD *)v12 == -1 && *(_QWORD *)(v12 + 8) == -4096 )
      goto LABEL_11;
    v13 = v10 + i;
    ++v10;
  }
  if ( v12 == v9 + 24 * v8 )
  {
LABEL_11:
    v16 = sub_27975D0(a1, a2, a3, a4);
    v17 = *(_DWORD *)(a1 + 176);
    v28 = a4;
    v29 = a2;
    v14 = v16;
    v30 = v16;
    if ( v17 )
    {
      v18 = 1;
      v20 = 0;
      for ( j = (v17 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned __int64)(37 * a4) << 32) | ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v17 - 1) & v24 )
      {
        v19 = *(_QWORD *)(a1 + 160);
        v22 = (unsigned int *)(v19 + 24LL * j);
        v23 = *v22;
        if ( a4 == *v22 && a2 == *((_QWORD *)v22 + 1) )
          break;
        if ( v23 == -1 )
        {
          if ( *((_QWORD *)v22 + 1) == -4096 )
          {
            v25 = *(_DWORD *)(a1 + 168);
            if ( v20 )
              v22 = v20;
            ++*(_QWORD *)(a1 + 152);
            v26 = v25 + 1;
            v27 = v22;
            if ( 4 * (v25 + 1) >= 3 * v17 )
              goto LABEL_33;
            if ( v17 - *(_DWORD *)(a1 + 172) - v26 <= v17 >> 3 )
              goto LABEL_34;
            goto LABEL_25;
          }
        }
        else if ( v23 == -2 && *((_QWORD *)v22 + 1) == -8192 && !v20 )
        {
          v20 = (unsigned int *)(v19 + 24LL * j);
        }
        v24 = v18 + j;
        ++v18;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 152);
      v27 = 0;
LABEL_33:
      v17 *= 2;
LABEL_34:
      sub_2797090(a1 + 152, v17);
      sub_2790630(a1 + 152, (int *)&v28, &v27);
      a4 = v28;
      v22 = v27;
      v26 = *(_DWORD *)(a1 + 168) + 1;
LABEL_25:
      *(_DWORD *)(a1 + 168) = v26;
      if ( *v22 != -1 || *((_QWORD *)v22 + 1) != -4096 )
        --*(_DWORD *)(a1 + 172);
      *v22 = a4;
      *((_QWORD *)v22 + 1) = v29;
      v22[4] = v30;
    }
  }
  else
  {
    return *(unsigned int *)(v12 + 16);
  }
  return v14;
}
