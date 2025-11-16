// Function: sub_1599A20
// Address: 0x1599a20
//
__int64 __fastcall sub_1599A20(__int64 **a1)
{
  __int64 v2; // rbx
  unsigned int v3; // esi
  __int64 v4; // rdi
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 ***v7; // rax
  __int64 **v8; // r9
  __int64 v9; // r13
  int v11; // r11d
  __int64 ***v12; // r14
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  __int64 **v16; // r12
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 **v21; // rsi
  int v22; // r9d
  __int64 ***v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r13d
  __int64 ***v29; // rdi
  __int64 **v30; // rcx

  v2 = **a1;
  v3 = *(_DWORD *)(v2 + 1672);
  v4 = v2 + 1648;
  if ( !v3 )
  {
    ++*(_QWORD *)(v2 + 1648);
    goto LABEL_19;
  }
  v5 = *(_QWORD *)(v2 + 1656);
  v6 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v7 = (__int64 ***)(v5 + 16LL * v6);
  v8 = *v7;
  if ( *v7 != a1 )
  {
    v11 = 1;
    v12 = 0;
    while ( v8 != (__int64 **)-8LL )
    {
      if ( !v12 && v8 == (__int64 **)-16LL )
        v12 = v7;
      v6 = (v3 - 1) & (v11 + v6);
      v7 = (__int64 ***)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == a1 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v7;
    v13 = *(_DWORD *)(v2 + 1664);
    ++*(_QWORD *)(v2 + 1648);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(v2 + 1668) - v14 > v3 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(v2 + 1664) = v14;
        if ( *v12 != (__int64 **)-8LL )
          --*(_DWORD *)(v2 + 1668);
        *v12 = a1;
        v12[1] = 0;
        goto LABEL_14;
      }
      sub_1599810(v4, v3);
      v24 = *(_DWORD *)(v2 + 1672);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(v2 + 1656);
        v27 = 1;
        v28 = v25 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v14 = *(_DWORD *)(v2 + 1664) + 1;
        v29 = 0;
        v12 = (__int64 ***)(v26 + 16LL * v28);
        v30 = *v12;
        if ( *v12 != a1 )
        {
          while ( v30 != (__int64 **)-8LL )
          {
            if ( !v29 && v30 == (__int64 **)-16LL )
              v29 = v12;
            v28 = v25 & (v27 + v28);
            v12 = (__int64 ***)(v26 + 16LL * v28);
            v30 = *v12;
            if ( *v12 == a1 )
              goto LABEL_11;
            ++v27;
          }
          if ( v29 )
            v12 = v29;
        }
        goto LABEL_11;
      }
LABEL_48:
      ++*(_DWORD *)(v2 + 1664);
      BUG();
    }
LABEL_19:
    sub_1599810(v4, 2 * v3);
    v17 = *(_DWORD *)(v2 + 1672);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v2 + 1656);
      v20 = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = *(_DWORD *)(v2 + 1664) + 1;
      v12 = (__int64 ***)(v19 + 16LL * v20);
      v21 = *v12;
      if ( *v12 != a1 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != (__int64 **)-8LL )
        {
          if ( !v23 && v21 == (__int64 **)-16LL )
            v23 = v12;
          v20 = v18 & (v22 + v20);
          v12 = (__int64 ***)(v19 + 16LL * v20);
          v21 = *v12;
          if ( *v12 == a1 )
            goto LABEL_11;
          ++v22;
        }
        if ( v23 )
          v12 = v23;
      }
      goto LABEL_11;
    }
    goto LABEL_48;
  }
LABEL_3:
  v9 = (__int64)v7[1];
  if ( v9 )
    return v9;
  v12 = v7;
LABEL_14:
  v15 = sub_1648A60(24, 0);
  v9 = v15;
  if ( v15 )
  {
    sub_1648CB0(v15, a1, 15);
    *(_DWORD *)(v9 + 20) &= 0xF0000000;
  }
  v16 = v12[1];
  v12[1] = (__int64 **)v9;
  if ( v16 )
  {
    sub_164BE60(v16);
    sub_1648B90(v16);
    return (__int64)v12[1];
  }
  return v9;
}
