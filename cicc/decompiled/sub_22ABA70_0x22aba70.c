// Function: sub_22ABA70
// Address: 0x22aba70
//
__int64 __fastcall sub_22ABA70(__int64 **a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rsi
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  char v17; // al
  __int64 *v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 48);
  v21 = a1[2][3];
  v3 = *a1[1];
  v4 = **a1;
  v5 = *(_QWORD *)(v4 + 40);
  if ( *(_BYTE *)(v2 + 84) )
  {
    v6 = *(_QWORD **)(v2 + 64);
    v7 = &v6[*(unsigned int *)(v2 + 76)];
    if ( v6 != v7 )
    {
      while ( v5 != *v6 )
      {
        if ( v7 == ++v6 )
          goto LABEL_8;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(v2 + 56, v5) )
  {
    return 0;
  }
LABEL_8:
  v9 = sub_D47930(v2);
  if ( !v9 )
    return 0;
  v14 = v9;
  if ( (unsigned __int8)sub_B19720(v21, v9, *(_QWORD *)(v4 + 40)) )
    goto LABEL_19;
  if ( *(_BYTE *)v4 != 84 || !v3 )
    return 0;
  v12 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 0 )
  {
    v15 = 0;
    v12 = 8LL * (unsigned int)v12;
    do
    {
      while ( 1 )
      {
        v10 = *(__int64 **)(v4 - 8);
        v16 = v10[v15 / 2];
        if ( v3 == v16 )
        {
          if ( v16 )
            break;
        }
        v15 += 8LL;
        if ( v15 == v12 )
          goto LABEL_19;
      }
      v20 = v12;
      v17 = sub_B19720(v21, v14, v10[4 * *(unsigned int *)(v4 + 72) + v15 / 8]);
      v12 = v20;
      if ( !v17 )
        return 0;
      v15 += 8LL;
    }
    while ( v15 != v20 );
  }
LABEL_19:
  v18 = a1[3];
  if ( *((_BYTE *)v18 + 108) )
  {
    v19 = (__int64 *)v18[11];
    v11 = *((unsigned int *)v18 + 25);
    v10 = &v19[v11];
    if ( v19 != v10 )
    {
      while ( v2 != *v19 )
      {
        if ( v10 == ++v19 )
          goto LABEL_26;
      }
      return 1;
    }
LABEL_26:
    if ( (unsigned int)v11 < *((_DWORD *)v18 + 24) )
    {
      *((_DWORD *)v18 + 25) = v11 + 1;
      *v10 = v2;
      ++v18[10];
      return 1;
    }
  }
  sub_C8CC70((__int64)(v18 + 10), v2, (__int64)v10, v11, v12, v13);
  return 1;
}
