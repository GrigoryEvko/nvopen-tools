// Function: sub_FCD9F0
// Address: 0xfcd9f0
//
__int64 __fastcall sub_FCD9F0(__int64 a1)
{
  size_t v1; // rdx
  __int64 v2; // r14
  char *v3; // r13
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rcx
  __int64 v17; // rcx

  v1 = 0;
  v2 = *(_QWORD *)a1;
  v3 = off_4C5D0B0[0];
  if ( off_4C5D0B0[0] )
    v1 = strlen(off_4C5D0B0[0]);
  v4 = (__int64)v3;
  result = sub_B91CC0(v2, v3, v1);
  if ( result )
  {
    v9 = *(unsigned __int8 *)(result - 16);
    if ( (v9 & 2) != 0 )
    {
      v10 = *(_QWORD *)(result - 32);
    }
    else
    {
      v9 = 8LL * (((unsigned __int8)v9 >> 2) & 0xF);
      v10 = result - v9 - 16;
    }
    v11 = *(_QWORD *)(*(_QWORD *)v10 + 136LL);
    v12 = *(_QWORD *)(v11 + 24);
    if ( *(_DWORD *)(v11 + 32) > 0x40u )
      v12 = *(_QWORD *)v12;
    v13 = *(_QWORD *)a1;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 1) != 0 )
    {
      sub_B2C6D0(*(_QWORD *)a1, v4, v9, v6);
      v14 = *(_QWORD *)(v13 + 96);
      result = 5LL * *(_QWORD *)(v13 + 104);
      v15 = v14 + 40LL * *(_QWORD *)(v13 + 104);
      if ( (*(_BYTE *)(v13 + 2) & 1) != 0 )
      {
        result = sub_B2C6D0(v13, v4, v9, v17);
        v14 = *(_QWORD *)(v13 + 96);
      }
    }
    else
    {
      v14 = *(_QWORD *)(v13 + 96);
      result = 5LL * *(_QWORD *)(v13 + 104);
      v15 = v14 + 40LL * *(_QWORD *)(v13 + 104);
    }
    for ( ; v14 != v15; ++*(_QWORD *)(a1 + 176) )
    {
      while ( 1 )
      {
        v16 = *(unsigned int *)(v14 + 32);
        result = (unsigned int)(1 << v16);
        if ( (v12 & result) != 0 )
          break;
LABEL_12:
        v14 += 40;
        if ( v14 == v15 )
          return result;
      }
      if ( !*(_BYTE *)(a1 + 204) )
        goto LABEL_22;
      result = *(_QWORD *)(a1 + 184);
      v16 = *(unsigned int *)(a1 + 196);
      v9 = result + 8 * v16;
      if ( result != v9 )
      {
        while ( *(_QWORD *)result != v14 )
        {
          result += 8;
          if ( v9 == result )
            goto LABEL_18;
        }
        goto LABEL_12;
      }
LABEL_18:
      if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 192) )
      {
LABEL_22:
        result = (__int64)sub_C8CC70(a1 + 176, v14, v9, v16, v7, v8);
        goto LABEL_12;
      }
      *(_DWORD *)(a1 + 196) = v16 + 1;
      *(_QWORD *)v9 = v14;
      v14 += 40;
    }
  }
  return result;
}
