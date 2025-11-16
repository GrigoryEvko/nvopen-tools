// Function: sub_159BBF0
// Address: 0x159bbf0
//
__int64 __fastcall sub_159BBF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // r14
  __int64 v8; // rdx
  int v9; // r10d
  __int64 *v10; // r11
  __int64 v11; // rdi
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r8
  unsigned int j; // eax
  __int64 *v15; // r15
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 result; // rax
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rax
  int v22; // edi
  int v23; // edi
  int v24; // r10d
  __int64 v25; // rsi
  __int64 *v26; // r9
  unsigned __int64 v27; // r8
  unsigned __int64 v28; // r8
  unsigned int i; // eax
  __int64 v30; // r8
  unsigned int v31; // eax
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 *v33; // [rsp+18h] [rbp-48h] BYREF
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  __int64 v35; // [rsp+28h] [rbp-38h]

  v35 = a2;
  v34 = a1;
  v4 = sub_15E0530(a1);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 1768LL);
  v7 = *(_QWORD *)v4 + 1744LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 1744);
LABEL_27:
    sub_159B4E0(v7, 2 * v6);
    v22 = *(_DWORD *)(v5 + 1768);
    if ( !v22 )
    {
      ++*(_DWORD *)(v5 + 1760);
      BUG();
    }
    v8 = v34;
    v23 = v22 - 1;
    v24 = 1;
    v26 = 0;
    v27 = (((((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)
           | ((unsigned __int64)(((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32)) >> 22)
        ^ ((((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)
          | ((unsigned __int64)(((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32));
    v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
        ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
    for ( i = v23 & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; i = v23 & v31 )
    {
      v25 = *(_QWORD *)(v5 + 1752);
      v15 = (__int64 *)(v25 + 24LL * i);
      v30 = *v15;
      if ( *v15 == v34 && v15[1] == v35 )
        break;
      if ( v30 == -8 )
      {
        if ( v15[1] == -8 )
        {
          v20 = *(_DWORD *)(v5 + 1760) + 1;
          if ( v26 )
            v15 = v26;
          goto LABEL_18;
        }
      }
      else if ( v30 == -16 && v15[1] == -16 && !v26 )
      {
        v26 = (__int64 *)(v25 + 24LL * i);
      }
      v31 = v24 + i;
      ++v24;
    }
    goto LABEL_37;
  }
  v8 = v34;
  v9 = 1;
  v10 = 0;
  v11 = *(_QWORD *)(v5 + 1752);
  v12 = (((((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)
         | ((unsigned __int64)(((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)
        | ((unsigned __int64)(((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4)) << 32));
  v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
      ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
  for ( j = (v6 - 1) & (((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - ((_DWORD)v13 << 27))); ; j = (v6 - 1) & v17 )
  {
    v15 = (__int64 *)(v11 + 24LL * j);
    v16 = *v15;
    if ( *v15 == v34 && v15[1] == v35 )
    {
      result = v15[2];
      if ( !result )
        goto LABEL_21;
      return result;
    }
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && v15[1] == -16 && !v10 )
      v10 = (__int64 *)(v11 + 24LL * j);
LABEL_9:
    v17 = v9 + j;
    ++v9;
  }
  if ( v15[1] != -8 )
    goto LABEL_9;
  v19 = *(_DWORD *)(v5 + 1760);
  if ( v10 )
    v15 = v10;
  ++*(_QWORD *)(v5 + 1744);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
    goto LABEL_27;
  if ( v6 - *(_DWORD *)(v5 + 1764) - v20 > v6 >> 3 )
    goto LABEL_18;
  sub_159B4E0(v7, v6);
  sub_15977E0(v7, &v34, &v33);
  v15 = v33;
  v8 = v34;
LABEL_37:
  v20 = *(_DWORD *)(v5 + 1760) + 1;
LABEL_18:
  *(_DWORD *)(v5 + 1760) = v20;
  if ( *v15 != -8 || v15[1] != -8 )
    --*(_DWORD *)(v5 + 1764);
  *v15 = v8;
  v21 = v35;
  v15[2] = 0;
  v15[1] = v21;
LABEL_21:
  result = sub_1648A60(24, 2);
  if ( result )
  {
    v32 = result;
    sub_1594D00(result, a1, a2);
    result = v32;
  }
  v15[2] = result;
  return result;
}
