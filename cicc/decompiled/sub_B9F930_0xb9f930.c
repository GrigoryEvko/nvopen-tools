// Function: sub_B9F930
// Address: 0xb9f930
//
__int64 __fastcall sub_B9F930(__int64 a1, _BYTE *a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // rcx
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  unsigned int v12; // esi
  _BYTE *v13; // rdx
  __int64 v14; // rdi
  _QWORD *v15; // r10
  int v16; // r14d
  _BYTE *v17; // r9
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // r11
  __int64 v21; // rsi
  __int64 *v22; // rbx
  __int64 result; // rax
  int v24; // eax
  int v25; // edx
  int v26; // eax
  int v27; // r9d
  _BYTE *v28; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v29; // [rsp+18h] [rbp-38h] BYREF

  v3 = (__int64 *)sub_BD5C60(a1, a2);
  v28 = sub_B9F650(v3, a2);
  v4 = *v3;
  v5 = *(_DWORD *)(v4 + 624);
  v6 = *(_QWORD *)(v4 + 608);
  if ( v5 )
  {
    v7 = *(_QWORD *)(a1 + 24);
    v8 = v5 - 1;
    v9 = (v5 - 1) & (((unsigned int)*(_QWORD *)(a1 + 24) >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v7 == *v10 )
    {
LABEL_3:
      *v10 = -8192;
      --*(_DWORD *)(v4 + 616);
      ++*(_DWORD *)(v4 + 620);
    }
    else
    {
      v26 = 1;
      while ( v11 != -4096 )
      {
        v27 = v26 + 1;
        v9 = v8 & (v26 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_3;
        v26 = v27;
      }
    }
  }
  sub_B91270(a1);
  *(_QWORD *)(a1 + 24) = 0;
  v12 = *(_DWORD *)(v4 + 624);
  if ( !v12 )
  {
    ++*(_QWORD *)(v4 + 600);
    v29 = 0;
    goto LABEL_23;
  }
  v13 = v28;
  v14 = *(_QWORD *)(v4 + 608);
  v15 = 0;
  v16 = 1;
  v17 = v28;
  v18 = (v12 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v19 = (_QWORD *)(v14 + 16LL * v18);
  v20 = *v19;
  if ( v28 != (_BYTE *)*v19 )
  {
    while ( v20 != -4096 )
    {
      if ( v20 == -8192 && !v15 )
        v15 = v19;
      v18 = (v12 - 1) & (v16 + v18);
      v19 = (_QWORD *)(v14 + 16LL * v18);
      v20 = *v19;
      if ( v28 == (_BYTE *)*v19 )
        goto LABEL_6;
      ++v16;
    }
    if ( !v15 )
      v15 = v19;
    v24 = *(_DWORD *)(v4 + 616);
    ++*(_QWORD *)(v4 + 600);
    v25 = v24 + 1;
    v29 = v15;
    if ( 4 * (v24 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(v4 + 620) - v25 > v12 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(v4 + 616) = v25;
        if ( *v15 != -4096 )
          --*(_DWORD *)(v4 + 620);
        v15[1] = 0;
        v22 = v15 + 1;
        *v15 = v17;
        v13 = v28;
        goto LABEL_21;
      }
LABEL_24:
      sub_B95C80(v4 + 600, v12);
      sub_B92630(v4 + 600, (__int64 *)&v28, &v29);
      v17 = v28;
      v15 = v29;
      v25 = *(_DWORD *)(v4 + 616) + 1;
      goto LABEL_18;
    }
LABEL_23:
    v12 *= 2;
    goto LABEL_24;
  }
LABEL_6:
  v21 = v19[1];
  v22 = v19 + 1;
  if ( v21 )
  {
    sub_BD84D0(a1, v21);
    sub_B91290(a1);
    return j_j___libc_free_0(a1, 32);
  }
LABEL_21:
  *(_QWORD *)(a1 + 24) = v13;
  result = sub_B96EF0(a1);
  *v22 = a1;
  return result;
}
