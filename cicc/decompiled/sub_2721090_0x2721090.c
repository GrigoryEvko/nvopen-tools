// Function: sub_2721090
// Address: 0x2721090
//
__int64 __fastcall sub_2721090(__int64 a1, __int64 a2, __int64 *j, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // r15
  __int64 *v11; // rax
  __int64 *v12; // rbx
  __int64 *i; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 24;
  v8 = a1 + 1608;
  LODWORD(result) = *(_DWORD *)(a1 + 112);
  v19 = a1 + 1608;
LABEL_2:
  while ( 1 )
  {
    if ( !(_DWORD)result )
    {
      sub_2720DF0(a1, a2, j, v8, a5, a6);
      result = *(unsigned int *)(a1 + 112);
      if ( !(_DWORD)result )
        return result;
    }
    j = *(__int64 **)(a1 + 104);
    a2 = (unsigned int)result;
    v10 = j[(unsigned int)result - 1];
    *(_DWORD *)(a1 + 112) = result - 1;
    if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
    {
      v11 = *(__int64 **)(v10 - 8);
      v12 = &v11[4 * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v12 = (__int64 *)v10;
      j = (__int64 *)(32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
      v11 = (__int64 *)(v10 - (_QWORD)j);
    }
    for ( i = v11; v12 != i; i += 4 )
    {
      a2 = *i;
      if ( *(_BYTE *)*i > 0x1Cu )
        sub_27207D0(a1, a2, (__int64)j, v8, a5);
    }
    if ( *(_BYTE *)v10 == 84 )
    {
      a2 = (__int64)v20;
      v20[0] = *(_QWORD *)(v10 + 40);
      v14 = sub_27204A0(v7, v20, (__int64)j, v8, a5, a6);
      if ( !*(_BYTE *)(v14 + 2) )
      {
        *(_BYTE *)(v14 + 2) = 1;
        v15 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 16LL);
        if ( v15 )
        {
          while ( 1 )
          {
            j = *(__int64 **)(v15 + 24);
            if ( (unsigned __int8)(*(_BYTE *)j - 30) <= 0xAu )
              break;
            v15 = *(_QWORD *)(v15 + 8);
            if ( !v15 )
            {
              LODWORD(result) = *(_DWORD *)(a1 + 112);
              goto LABEL_2;
            }
          }
LABEL_16:
          a2 = (__int64)v20;
          v20[0] = j[5];
          v16 = sub_27204A0(v7, v20, (__int64)j, v8, a5, a6);
          if ( *(_BYTE *)(v16 + 3) )
            goto LABEL_17;
          *(_BYTE *)(v16 + 3) = 1;
          a2 = v20[0];
          if ( !*(_BYTE *)(a1 + 1636) )
            goto LABEL_27;
          v17 = *(__int64 **)(a1 + 1616);
          v18 = *(unsigned int *)(a1 + 1628);
          for ( j = &v17[v18]; j != v17; ++v17 )
          {
            if ( v20[0] == *v17 )
              goto LABEL_17;
          }
          if ( (unsigned int)v18 < *(_DWORD *)(a1 + 1624) )
          {
            *(_DWORD *)(a1 + 1628) = v18 + 1;
            *j = a2;
            ++*(_QWORD *)(a1 + 1608);
          }
          else
          {
LABEL_27:
            sub_C8CC70(v19, v20[0], (__int64)j, v8, a5, a6);
          }
LABEL_17:
          while ( 1 )
          {
            v15 = *(_QWORD *)(v15 + 8);
            if ( !v15 )
              break;
            j = *(__int64 **)(v15 + 24);
            if ( (unsigned __int8)(*(_BYTE *)j - 30) <= 0xAu )
              goto LABEL_16;
          }
        }
      }
    }
    LODWORD(result) = *(_DWORD *)(a1 + 112);
  }
}
