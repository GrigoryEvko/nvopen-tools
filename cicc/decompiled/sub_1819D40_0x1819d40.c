// Function: sub_1819D40
// Address: 0x1819d40
//
__int64 __fastcall sub_1819D40(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 v6; // rsi
  __int64 v7; // rdi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 *v13; // r13
  int v14; // edx
  _QWORD *v15; // r14
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r9
  unsigned int v22; // esi
  _QWORD *v23; // rax
  _BYTE *v24; // rsi
  int v25; // r10d
  int v26; // eax
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // [rsp+0h] [rbp-B0h]
  __int64 v32; // [rsp+8h] [rbp-A8h] BYREF
  char v33[16]; // [rsp+10h] [rbp-A0h] BYREF
  __int16 v34; // [rsp+20h] [rbp-90h]
  __int64 v35[16]; // [rsp+30h] [rbp-80h] BYREF

  v2 = a2;
  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  v32 = a2;
  if ( v5 <= 0x17u && v5 != 17 )
    return *(_QWORD *)(*a1 + 200LL);
  v6 = *((unsigned int *)a1 + 38);
  if ( !(_DWORD)v6 )
  {
    ++a1[16];
    goto LABEL_9;
  }
  v7 = a1[17];
  v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v4 != *v9 )
  {
    v25 = 1;
    v13 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v13 )
        v13 = v9;
      v8 = (v6 - 1) & (v25 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v4 == *v9 )
        goto LABEL_5;
      ++v25;
    }
    if ( !v13 )
      v13 = v9;
    v26 = *((_DWORD *)a1 + 36);
    ++a1[16];
    v14 = v26 + 1;
    if ( 4 * (v26 + 1) < (unsigned int)(3 * v6) )
    {
      if ( (int)v6 - *((_DWORD *)a1 + 37) - v14 > (unsigned int)v6 >> 3 )
        goto LABEL_11;
      goto LABEL_10;
    }
LABEL_9:
    LODWORD(v6) = 2 * v6;
LABEL_10:
    sub_176F940((__int64)(a1 + 16), v6);
    v6 = (__int64)&v32;
    sub_176A9A0((__int64)(a1 + 16), &v32, v35);
    v13 = (__int64 *)v35[0];
    v2 = v32;
    v14 = *((_DWORD *)a1 + 36) + 1;
LABEL_11:
    *((_DWORD *)a1 + 36) = v14;
    if ( *v13 != -8 )
      --*((_DWORD *)a1 + 37);
    v13[1] = 0;
    v15 = v13 + 1;
    *v13 = v2;
    v4 = v32;
    goto LABEL_14;
  }
LABEL_5:
  v11 = v9[1];
  if ( v11 )
    return v11;
  v15 = v9 + 1;
  v13 = v9;
LABEL_14:
  if ( *(_BYTE *)(v4 + 16) != 17 )
  {
    v11 = *(_QWORD *)(*a1 + 200LL);
    v13[1] = v11;
    return v11;
  }
  if ( *((_BYTE *)a1 + 100) )
    return *(_QWORD *)(*a1 + 200LL);
  v16 = *((_DWORD *)a1 + 24);
  if ( !v16 )
  {
    v27 = a1[1];
    v28 = *(_DWORD *)(v4 + 32) + (unsigned int)(*(_QWORD *)(v27 + 96) >> 1);
    if ( (*(_BYTE *)(v27 + 18) & 1) != 0 )
      sub_15E08E0(a1[1], v6);
    v29 = *(_QWORD *)(v27 + 88);
    if ( (_DWORD)v28 )
      v29 += 40 * v28;
    v13[1] = v29;
    goto LABEL_23;
  }
  if ( v16 == 1 )
  {
    v17 = a1[13];
    v18 = *a1;
    if ( !v17 )
    {
      v17 = *(_QWORD *)(v18 + 224);
      if ( v17 )
      {
        a1[13] = v17;
      }
      else
      {
        v17 = sub_18179B0(a1);
        v18 = *a1;
      }
    }
    if ( *(_QWORD *)(v18 + 224) )
    {
      v19 = *(_QWORD *)(a1[1] + 80LL);
      if ( !v19 )
        BUG();
      v20 = *(_QWORD *)(v19 + 24);
      v21 = v20 - 24;
      if ( v20 )
        goto LABEL_22;
    }
    else
    {
      v30 = *(_QWORD *)(v17 + 32);
      if ( v30 != *(_QWORD *)(v17 + 40) + 40LL && v30 )
      {
        v21 = v30 - 24;
LABEL_22:
        v31 = v21;
        sub_17CE510((__int64)v35, v21, 0, 0, 0);
        v22 = *(_DWORD *)(v4 + 32);
        v34 = 257;
        v23 = sub_1817AE0(a1, v22, v31);
        v13[1] = (__int64)sub_156E5B0(v35, (__int64)v23, (__int64)v33);
        sub_17CD270(v35);
        goto LABEL_23;
      }
    }
    v21 = 0;
    goto LABEL_22;
  }
LABEL_23:
  v24 = (_BYTE *)a1[32];
  if ( v24 == (_BYTE *)a1[33] )
  {
    sub_1287830((__int64)(a1 + 31), v24, v15);
    return v13[1];
  }
  else
  {
    if ( v24 )
    {
      *(_QWORD *)v24 = v13[1];
      v24 = (_BYTE *)a1[32];
    }
    a1[32] = v24 + 8;
    return v13[1];
  }
}
