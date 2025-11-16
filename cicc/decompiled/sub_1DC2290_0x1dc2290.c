// Function: sub_1DC2290
// Address: 0x1dc2290
//
__int64 __fastcall sub_1DC2290(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r8
  unsigned __int64 i; // rbx
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 result; // rax
  __int64 v11; // r12
  unsigned int v12; // esi
  char v13; // al
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // rbx
  char v17; // al
  __int64 v18; // rdx
  unsigned __int8 *v19; // rax
  unsigned int v20; // esi
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v6 = a3;
  for ( i = a2; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v8 = *(_QWORD *)(a2 + 24) + 24LL;
  do
  {
    v9 = *(_QWORD *)(i + 32);
    result = 5LL * *(unsigned int *)(i + 40);
    v11 = v9 + 40LL * *(unsigned int *)(i + 40);
    if ( v9 != v11 )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v8 == i )
      break;
  }
  while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  if ( v9 != v11 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v9 )
      {
        if ( *(_BYTE *)v9 == 12 )
        {
          v21 = v9;
          v14 = v9 + 40;
          v26 = v6;
          sub_1DC1EF0((__int64)a1, v21, v6, a4, v6, a6);
          v6 = v26;
          result = v11;
          if ( v14 == v11 )
            goto LABEL_18;
          goto LABEL_32;
        }
      }
      else if ( (*(_BYTE *)(v9 + 4) & 8) == 0 )
      {
        v12 = *(_DWORD *)(v9 + 8);
        if ( (int)v12 > 0 )
        {
          v13 = *(_BYTE *)(v9 + 3);
          if ( (v13 & 0x10) != 0 )
          {
            v22 = *(unsigned int *)(v6 + 8);
            v23 = v12;
            if ( (unsigned int)v22 >= *(_DWORD *)(v6 + 12) )
            {
              v27 = v6;
              sub_16CD150(v6, (const void *)(v6 + 16), 0, 16, v6, a6);
              v6 = v27;
              v23 = v12;
              v22 = *(unsigned int *)(v27 + 8);
            }
            v24 = (_QWORD *)(*(_QWORD *)v6 + 16 * v22);
            *v24 = v23;
            v24[1] = v9;
            ++*(_DWORD *)(v6 + 8);
          }
          else if ( (v13 & 0x40) != 0 )
          {
            v25 = v6;
            sub_1DC1CF0((__int64)a1, v12);
            v6 = v25;
          }
        }
      }
      v14 = v9 + 40;
      result = v11;
      if ( v14 == v11 )
      {
LABEL_18:
        while ( 1 )
        {
          i = *(_QWORD *)(i + 8);
          if ( v8 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
            break;
          v11 = *(_QWORD *)(i + 32);
          result = v11 + 40LL * *(unsigned int *)(i + 40);
          if ( v11 != result )
            goto LABEL_33;
        }
        v9 = v11;
        v11 = result;
        if ( v9 == result )
          break;
      }
      else
      {
LABEL_32:
        v11 = v14;
LABEL_33:
        v9 = v11;
        v11 = result;
      }
    }
  }
  v15 = *(_QWORD *)v6;
  v16 = *(_QWORD *)v6 + 16LL * *(unsigned int *)(v6 + 8);
  if ( *(_QWORD *)v6 != v16 )
  {
    do
    {
      while ( 1 )
      {
        v19 = *(unsigned __int8 **)(v15 + 8);
        v20 = *(_DWORD *)v15;
        v18 = *v19;
        if ( (_BYTE)v18 )
          break;
        v18 = v19[3];
        v17 = (unsigned __int8)v18 >> 4;
        LOBYTE(v18) = (unsigned __int8)v18 >> 6;
        result = v17 & 1;
        if ( ((unsigned __int8)result & (unsigned __int8)v18) == 0 )
          goto LABEL_23;
LABEL_24:
        v15 += 16;
        if ( v16 == v15 )
          return result;
      }
      if ( (_BYTE)v18 != 12
        || (v18 = v20 >> 5,
            result = *(unsigned int *)(*((_QWORD *)v19 + 3) + 4 * v18),
            _bittest((const int *)&result, v20)) )
      {
LABEL_23:
        result = sub_1DC1BF0(a1, v20, v18, a4, v6, a6);
        goto LABEL_24;
      }
      v15 += 16;
    }
    while ( v16 != v15 );
  }
  return result;
}
