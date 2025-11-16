// Function: sub_18DEC30
// Address: 0x18dec30
//
unsigned __int64 __fastcall sub_18DEC30(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 *v5; // r8
  __int64 *v6; // rsi
  unsigned __int64 result; // rax
  char v8; // dl
  unsigned __int8 *i; // r12
  char v10; // dl
  __int64 v11; // rax
  unsigned __int8 **v12; // rdi
  unsigned int v13; // r9d
  unsigned __int8 **v14; // rcx
  __int64 *v15; // rcx
  unsigned int v16; // edi
  __int64 *v17; // rdx

  v3 = a1 + 1152;
  v5 = *(__int64 **)(a1 + 1168);
  v6 = *(__int64 **)(a1 + 1160);
  do
  {
    if ( v5 != v6 )
    {
LABEL_3:
      result = (unsigned __int64)sub_16CCBA0(v3, a2);
      v5 = *(__int64 **)(a1 + 1168);
      v6 = *(__int64 **)(a1 + 1160);
      if ( !v8 )
        return result;
      goto LABEL_4;
    }
    v15 = &v6[*(unsigned int *)(a1 + 1180)];
    v16 = *(_DWORD *)(a1 + 1180);
    if ( v15 == v6 )
      goto LABEL_31;
    v17 = 0;
    do
    {
      result = *v6;
      if ( a2 == *v6 )
        return result;
      if ( result == -2 )
        v17 = v6;
      ++v6;
    }
    while ( v15 != v6 );
    if ( !v17 )
    {
LABEL_31:
      if ( v16 >= *(_DWORD *)(a1 + 1176) )
        goto LABEL_3;
      *(_DWORD *)(a1 + 1180) = v16 + 1;
      *v15 = a2;
      v6 = *(__int64 **)(a1 + 1160);
      ++*(_QWORD *)(a1 + 1152);
      v5 = *(__int64 **)(a1 + 1168);
    }
    else
    {
      *v17 = a2;
      v5 = *(__int64 **)(a1 + 1168);
      --*(_DWORD *)(a1 + 1184);
      v6 = *(__int64 **)(a1 + 1160);
      ++*(_QWORD *)(a1 + 1152);
    }
LABEL_4:
    for ( i = *(unsigned __int8 **)(a2 - 8LL * *(unsigned int *)(a2 + 8)); ; i = (unsigned __int8 *)v11 )
    {
      if ( v5 != v6 )
        goto LABEL_5;
      result = *(unsigned int *)(a1 + 1180);
      v12 = (unsigned __int8 **)&v5[result];
      v13 = *(_DWORD *)(a1 + 1180);
      if ( v12 != (unsigned __int8 **)v5 )
        break;
LABEL_20:
      if ( v13 < *(_DWORD *)(a1 + 1176) )
      {
        *(_DWORD *)(a1 + 1180) = v13 + 1;
        *v12 = i;
        v6 = *(__int64 **)(a1 + 1160);
        ++*(_QWORD *)(a1 + 1152);
        v5 = *(__int64 **)(a1 + 1168);
        goto LABEL_6;
      }
LABEL_5:
      result = (unsigned __int64)sub_16CCBA0(v3, (__int64)i);
      v5 = *(__int64 **)(a1 + 1168);
      v6 = *(__int64 **)(a1 + 1160);
      if ( !v10 )
        goto LABEL_15;
LABEL_6:
      if ( *i == 17 )
        goto LABEL_15;
      v11 = sub_15B0BB0(i);
      v5 = *(__int64 **)(a1 + 1168);
      v6 = *(__int64 **)(a1 + 1160);
    }
    result = (unsigned __int64)v5;
    v14 = 0;
    while ( i != *(unsigned __int8 **)result )
    {
      if ( *(_QWORD *)result == -2 )
        v14 = (unsigned __int8 **)result;
      result += 8LL;
      if ( v12 == (unsigned __int8 **)result )
      {
        if ( !v14 )
          goto LABEL_20;
        *v14 = i;
        v5 = *(__int64 **)(a1 + 1168);
        --*(_DWORD *)(a1 + 1184);
        v6 = *(__int64 **)(a1 + 1160);
        ++*(_QWORD *)(a1 + 1152);
        goto LABEL_6;
      }
    }
LABEL_15:
    if ( *(_DWORD *)(a2 + 8) != 2 )
      break;
    a2 = *(_QWORD *)(a2 - 8);
  }
  while ( a2 );
  return result;
}
