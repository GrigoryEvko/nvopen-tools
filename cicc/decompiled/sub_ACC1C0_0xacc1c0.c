// Function: sub_ACC1C0
// Address: 0xacc1c0
//
__int64 __fastcall sub_ACC1C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdx
  int v7; // r10d
  __int64 *v8; // rcx
  unsigned int i; // eax
  __int64 *v10; // rdi
  __int64 v11; // r9
  unsigned int v12; // eax
  __int64 *v13; // rbx
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // ecx
  int v20; // r11d
  __int64 *v21; // r10
  __int64 v22; // rsi
  int v23; // r9d
  unsigned int j; // eax
  __int64 v25; // r8
  unsigned int v26; // eax
  int v27; // eax
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 *v30; // [rsp+18h] [rbp-38h] BYREF
  __int64 v31; // [rsp+20h] [rbp-30h] BYREF
  __int64 v32; // [rsp+28h] [rbp-28h]

  v4 = *(_QWORD *)sub_B2BE50(a1);
  v31 = a1;
  v32 = a2;
  v5 = *(_DWORD *)(v4 + 2016);
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 1992);
    v30 = 0;
    goto LABEL_28;
  }
  v6 = *(_QWORD *)(v4 + 2000);
  v7 = 1;
  v8 = 0;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
              | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v5 - 1) & v12 )
  {
    v10 = (__int64 *)(v6 + 24LL * i);
    v11 = *v10;
    if ( a1 == *v10 && a2 == v10[1] )
    {
      v13 = v10 + 2;
      goto LABEL_12;
    }
    if ( v11 == -4096 )
      break;
    if ( v11 == -8192 && v10[1] == -8192 && !v8 )
      v8 = (__int64 *)(v6 + 24LL * i);
LABEL_9:
    v12 = v7 + i;
    ++v7;
  }
  if ( v10[1] != -4096 )
    goto LABEL_9;
  v15 = *(_DWORD *)(v4 + 2008);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(v4 + 1992);
  v16 = v15 + 1;
  v30 = v8;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_28:
    sub_ACBB00(v4 + 1992, 2 * v5);
    v19 = *(_DWORD *)(v4 + 2016);
    if ( v19 )
    {
      v17 = v31;
      v20 = 1;
      v21 = 0;
      v23 = v19 - 1;
      for ( j = (v19 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4)
                  | ((unsigned __int64)(((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4)))); ; j = v23 & v26 )
      {
        v22 = *(_QWORD *)(v4 + 2000);
        v8 = (__int64 *)(v22 + 24LL * j);
        v25 = *v8;
        if ( *v8 == v31 && v8[1] == v32 )
          break;
        if ( v25 == -4096 )
        {
          if ( v8[1] == -4096 )
          {
            if ( v21 )
              v8 = v21;
            break;
          }
        }
        else if ( v25 == -8192 && v8[1] == -8192 && !v21 )
        {
          v21 = (__int64 *)(v22 + 24LL * j);
        }
        v26 = v20 + j;
        ++v20;
      }
      v28 = *(_DWORD *)(v4 + 2008);
      v30 = v8;
      v16 = v28 + 1;
    }
    else
    {
      v27 = *(_DWORD *)(v4 + 2008);
      v17 = v31;
      v30 = 0;
      v8 = 0;
      v16 = v27 + 1;
    }
    goto LABEL_22;
  }
  v17 = a1;
  if ( v5 - *(_DWORD *)(v4 + 2012) - v16 <= v5 >> 3 )
  {
    sub_ACBB00(v4 + 1992, v5);
    sub_AC6F30(v4 + 1992, &v31, &v30);
    v17 = v31;
    v8 = v30;
    v16 = *(_DWORD *)(v4 + 2008) + 1;
  }
LABEL_22:
  *(_DWORD *)(v4 + 2008) = v16;
  if ( *v8 != -4096 || v8[1] != -4096 )
    --*(_DWORD *)(v4 + 2012);
  *v8 = v17;
  v18 = v32;
  v13 = v8 + 2;
  v8[2] = 0;
  v8[1] = v18;
LABEL_12:
  result = *v13;
  if ( !*v13 )
  {
    result = sub_BD2C40(24, unk_3F289A0);
    if ( result )
    {
      v29 = result;
      sub_AC3E90(result, a1, a2);
      result = v29;
    }
    *v13 = result;
  }
  return result;
}
