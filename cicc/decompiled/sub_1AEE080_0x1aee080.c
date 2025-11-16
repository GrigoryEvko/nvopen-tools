// Function: sub_1AEE080
// Address: 0x1aee080
//
__int64 __fastcall sub_1AEE080(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r15d
  __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r14
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rax
  bool v13; // zf
  __int64 *v14; // r8
  __int64 *v15; // rdi
  int v16; // r15d
  int v17; // eax
  int v18; // r8d
  __int64 *v19; // rcx
  unsigned int i; // r13d
  __int64 v21; // rdi
  __int64 *v22; // rdx
  __int64 v23; // rsi
  char v24; // al
  unsigned int v25; // r13d
  int v26; // [rsp+14h] [rbp-4Ch]
  __int64 *v27; // [rsp+18h] [rbp-48h]
  __int64 *v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v30[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v8 = *(_DWORD *)(*a2 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(*a2 + 23) & 0x40) != 0 )
      v9 = *(_QWORD *)(v6 - 8);
    else
      v9 = v6 - 24 * v8;
    v10 = *(_QWORD *)(a1 + 8);
    v27 = (__int64 *)*a2;
    v11 = (_QWORD *)(v9 + 24LL * *(unsigned int *)(v6 + 56) + 8);
    v12 = sub_1AEDC40(v11, (__int64)&v11[v8]);
    v13 = (*(_BYTE *)(v6 + 23) & 0x40) == 0;
    v14 = v27;
    v30[0] = v12;
    if ( v13 )
    {
      v15 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    }
    else
    {
      v15 = *(__int64 **)(v6 - 8);
      v14 = &v15[3 * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)];
    }
    v16 = v4 - 1;
    v29 = sub_18FDB50(v15, v14);
    v17 = sub_1AED750(&v29, (__int64 *)v30);
    v18 = 1;
    v19 = 0;
    for ( i = v16 & v17; ; i = v16 & v25 )
    {
      v21 = *a2;
      v22 = (__int64 *)(v10 + 8LL * i);
      v23 = *v22;
      if ( *v22 == -8 || *a2 == -8 || *a2 == -16 || v23 == -16 )
      {
        if ( v23 == v21 )
        {
LABEL_17:
          *a3 = v22;
          return 1;
        }
      }
      else
      {
        v26 = v18;
        v28 = v19;
        v24 = sub_15F41F0(v21, v23);
        v19 = v28;
        v18 = v26;
        v22 = (__int64 *)(v10 + 8LL * i);
        if ( v24 )
          goto LABEL_17;
        v23 = *(_QWORD *)(v10 + 8LL * i);
      }
      if ( v23 == -8 )
        break;
      if ( v23 == -16 && !v19 )
        v19 = v22;
      v25 = v18 + i;
      ++v18;
    }
    if ( !v19 )
      v19 = v22;
    *a3 = v19;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
