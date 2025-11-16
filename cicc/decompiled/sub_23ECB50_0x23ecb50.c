// Function: sub_23ECB50
// Address: 0x23ecb50
//
__int64 __fastcall sub_23ECB50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // rcx
  __int64 *v8; // r14
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rdi
  int v16; // edx
  __int16 v17; // ax
  __int64 *v18; // rdi
  __int64 v19; // rax
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rsi
  int v25; // r9d
  __int64 *v26; // r8
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  __int64 *v30; // rdi
  unsigned int v31; // r13d
  int v32; // r8d
  __int64 v33; // rcx
  _QWORD v34[8]; // [rsp+0h] [rbp-40h] BYREF

  v4 = a1 + 1032;
  v5 = *(_DWORD *)(a1 + 1056);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 1032);
    goto LABEL_35;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 1040);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return *((unsigned __int8 *)v10 + 8);
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return *((unsigned __int8 *)v10 + 8);
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 1048);
  ++*(_QWORD *)(a1 + 1032);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_35:
    sub_23EC970(v4, 2 * v5);
    v20 = *(_DWORD *)(a1 + 1056);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 1040);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 1048) + 1;
      v8 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v8;
      if ( a2 != *v8 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( v24 == -8192 && !v26 )
            v26 = v8;
          v23 = v21 & (v25 + v23);
          v8 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v8;
          if ( a2 == *v8 )
            goto LABEL_14;
          ++v25;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_14;
    }
    goto LABEL_60;
  }
  if ( v5 - *(_DWORD *)(a1 + 1052) - v14 <= v5 >> 3 )
  {
    sub_23EC970(v4, v5);
    v27 = *(_DWORD *)(a1 + 1056);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 1040);
      v30 = 0;
      v31 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 1;
      v14 = *(_DWORD *)(a1 + 1048) + 1;
      v8 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v8;
      if ( a2 != *v8 )
      {
        while ( v33 != -4096 )
        {
          if ( !v30 && v33 == -8192 )
            v30 = v8;
          v31 = v28 & (v32 + v31);
          v8 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v8;
          if ( a2 == *v8 )
            goto LABEL_14;
          ++v32;
        }
        if ( v30 )
          v8 = v30;
      }
      goto LABEL_14;
    }
LABEL_60:
    ++*(_DWORD *)(a1 + 1048);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 1048) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 1052);
  *v8 = a2;
  *((_BYTE *)v8 + 8) = 0;
  v15 = *(_QWORD *)(a2 + 72);
  v16 = *(unsigned __int8 *)(v15 + 8);
  if ( ((_BYTE)v16 == 12
     || (unsigned __int8)v16 <= 3u
     || (_BYTE)v16 == 5
     || (v16 & 0xFD) == 4
     || (v16 & 0xFB) == 0xA
     || ((unsigned __int8)(*(_BYTE *)(v15 + 8) - 15) <= 3u || v16 == 20) && (unsigned __int8)sub_BCEBA0(v15, 0))
    && (!sub_B4D040(a2) || (v19 = sub_B43CC0(a2), sub_B4CED0((__int64)v34, a2, v19), v34[0]))
    && (!(_BYTE)qword_4FE0C48 || !(unsigned __int8)sub_2A4D8A0(a2, 0))
    && (v17 = *(_WORD *)(a2 + 2), (v17 & 0x40) == 0)
    && (v17 & 0x80u) == 0 )
  {
    v18 = *(__int64 **)(a1 + 1024);
    result = 1;
    if ( v18 )
      result = (unsigned int)sub_D90430(v18, a2) ^ 1;
  }
  else
  {
    result = 0;
  }
  *((_BYTE *)v8 + 8) = result;
  return result;
}
