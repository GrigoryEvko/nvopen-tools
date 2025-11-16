// Function: sub_1467C40
// Address: 0x1467c40
//
_QWORD *__fastcall sub_1467C40(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // r13d
  int v5; // esi
  _QWORD *v6; // r12
  int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rsi
  char v13; // r13
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // r9
  int v21; // r10d
  __int64 v22; // [rsp+8h] [rbp-98h]
  void *v23; // [rsp+10h] [rbp-90h] BYREF
  char v24[16]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  _QWORD *v26; // [rsp+40h] [rbp-60h] BYREF
  _BYTE v27[16]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v28; // [rsp+58h] [rbp-48h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_3;
  }
  v8 = v4 - 1;
  v22 = *(_QWORD *)(a1 + 8);
  sub_1457D90(&v23, -8, 0);
  sub_1457D90(&v26, -16, 0);
  v9 = a2[3];
  v10 = v8 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v6 = (_QWORD *)(v22 + 48LL * v10);
  v11 = v6[3];
  if ( v9 == v11 )
  {
    v12 = v28;
    v13 = 1;
  }
  else
  {
    v12 = v28;
    v20 = (_QWORD *)(v22 + 48LL * (v8 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))));
    v21 = 1;
    v6 = 0;
    while ( v25 != v11 )
    {
      if ( v28 != v11 || v6 )
        v20 = v6;
      v10 = v8 & (v21 + v10);
      v6 = (_QWORD *)(v22 + 48LL * v10);
      v11 = v6[3];
      if ( v9 == v11 )
      {
        v13 = 1;
        goto LABEL_7;
      }
      ++v21;
      v6 = v20;
      v20 = (_QWORD *)(v22 + 48LL * v10);
    }
    v13 = 0;
    if ( !v6 )
      v6 = v20;
  }
LABEL_7:
  v26 = &unk_49EE2B0;
  if ( v12 != 0 && v12 != -8 && v12 != -16 )
    sub_1649B30(v27);
  v23 = &unk_49EE2B0;
  if ( v25 != 0 && v25 != -8 && v25 != -16 )
    sub_1649B30(v24);
  if ( !v13 )
  {
    v15 = *(_DWORD *)(a1 + 16);
    v4 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)a1;
    v7 = v15 + 1;
    if ( 4 * v7 < 3 * v4 )
    {
      if ( v4 - (v7 + *(_DWORD *)(a1 + 20)) > v4 >> 3 )
        goto LABEL_17;
      v5 = v4;
LABEL_4:
      sub_14676C0(a1, v5);
      sub_145F6E0(a1, (__int64)a2, &v26);
      v6 = v26;
      v7 = *(_DWORD *)(a1 + 16) + 1;
LABEL_17:
      *(_DWORD *)(a1 + 16) = v7;
      sub_1457D90(&v26, -8, 0);
      v16 = v28;
      if ( v28 != v6[3] )
        --*(_DWORD *)(a1 + 20);
      v26 = &unk_49EE2B0;
      if ( v16 != 0 && v16 != -8 && v16 != -16 )
        sub_1649B30(v27);
      v17 = v6[3];
      v18 = a2[3];
      if ( v17 != v18 )
      {
        if ( v17 != 0 && v17 != -8 && v17 != -16 )
        {
          sub_1649B30(v6 + 1);
          v18 = a2[3];
        }
        v6[3] = v18;
        if ( v18 != -8 && v18 != 0 && v18 != -16 )
          sub_1649AC0(v6 + 1, a2[1] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v19 = a2[4];
      v6[5] = 0;
      v6[4] = v19;
      return v6;
    }
LABEL_3:
    v5 = 2 * v4;
    goto LABEL_4;
  }
  return v6;
}
