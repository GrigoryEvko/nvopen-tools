// Function: sub_13C95C0
// Address: 0x13c95c0
//
__int64 __fastcall sub_13C95C0(__int64 **a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // r15
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 *v22; // rdi
  _QWORD *v23; // rax
  _QWORD *v24; // rsi
  unsigned int v25; // r8d
  _QWORD *v26; // rcx
  char v27; // al
  _QWORD *v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 48);
  v3 = *(_QWORD **)(v2 + 72);
  v31 = a1[2][3];
  v4 = *a1[1];
  v5 = **a1;
  v6 = *(_QWORD **)(v2 + 64);
  v7 = *(_QWORD *)(v5 + 40);
  if ( v3 == v6 )
  {
    v8 = &v6[*(unsigned int *)(v2 + 84)];
    if ( v6 == v8 )
    {
      v28 = *(_QWORD **)(v2 + 64);
    }
    else
    {
      do
      {
        if ( v7 == *v6 )
          break;
        ++v6;
      }
      while ( v8 != v6 );
      v28 = v8;
    }
  }
  else
  {
    v30 = *(_QWORD *)(v5 + 40);
    v8 = &v3[*(unsigned int *)(v2 + 80)];
    v6 = (_QWORD *)sub_16CC9F0(v2 + 56, v7);
    if ( v30 == *v6 )
    {
      v20 = *(_QWORD *)(v2 + 72);
      if ( v20 == *(_QWORD *)(v2 + 64) )
        v21 = *(unsigned int *)(v2 + 84);
      else
        v21 = *(unsigned int *)(v2 + 80);
      v28 = (_QWORD *)(v20 + 8 * v21);
    }
    else
    {
      v9 = *(_QWORD *)(v2 + 72);
      if ( v9 != *(_QWORD *)(v2 + 64) )
      {
        v6 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(v2 + 80));
        goto LABEL_5;
      }
      v6 = (_QWORD *)(v9 + 8LL * *(unsigned int *)(v2 + 84));
      v28 = v6;
    }
  }
  while ( v28 != v6 && *v6 >= 0xFFFFFFFFFFFFFFFELL )
    ++v6;
LABEL_5:
  if ( v6 != v8 )
    return 0;
  v11 = sub_13FCB50(v2);
  if ( !v11 )
    return 0;
  v15 = v11;
  if ( !(unsigned __int8)sub_15CC8F0(v31, v11, *(_QWORD *)(v5 + 40), v12, v13) )
  {
    if ( *(_BYTE *)(v5 + 16) == 77 && v4 )
    {
      if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) != 0 )
      {
        v16 = 0;
        v17 = 8LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        while ( 1 )
        {
          v18 = (*(_BYTE *)(v5 + 23) & 0x40) != 0 ? *(_QWORD *)(v5 - 8) : v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
          v19 = *(_QWORD *)(v18 + 3 * v16);
          if ( v4 == v19 )
          {
            if ( v19 )
            {
              v29 = v17;
              v27 = sub_15CC8F0(v31, v15, *(_QWORD *)(v16 + v18 + 24LL * *(unsigned int *)(v5 + 56) + 8), v14, v17);
              v17 = v29;
              if ( !v27 )
                return 0;
            }
          }
          v16 += 8;
          if ( v17 == v16 )
            goto LABEL_31;
        }
      }
      goto LABEL_31;
    }
    return 0;
  }
LABEL_31:
  v22 = a1[3];
  v23 = (_QWORD *)v22[11];
  if ( (_QWORD *)v22[12] != v23 )
    goto LABEL_32;
  v24 = &v23[*((unsigned int *)v22 + 27)];
  v25 = *((_DWORD *)v22 + 27);
  if ( v23 != v24 )
  {
    v26 = 0;
    while ( v2 != *v23 )
    {
      if ( *v23 == -2 )
        v26 = v23;
      if ( v24 == ++v23 )
      {
        if ( !v26 )
          goto LABEL_45;
        *v26 = v2;
        --*((_DWORD *)v22 + 28);
        ++v22[10];
        return 1;
      }
    }
    return 1;
  }
LABEL_45:
  if ( v25 < *((_DWORD *)v22 + 26) )
  {
    *((_DWORD *)v22 + 27) = v25 + 1;
    *v24 = v2;
    ++v22[10];
  }
  else
  {
LABEL_32:
    sub_16CCBA0(v22 + 10, v2);
  }
  return 1;
}
