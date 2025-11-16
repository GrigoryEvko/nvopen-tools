// Function: sub_18DFCA0
// Address: 0x18dfca0
//
__int64 __fastcall sub_18DFCA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // r13
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // r12
  __int64 *i; // r14
  __int64 v15; // rax
  __int64 v16; // r12
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdi
  __int64 *v22; // rcx
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 24;
  LODWORD(result) = *(_DWORD *)(a1 + 120);
  v23 = a1 + 1616;
  while ( 1 )
  {
    if ( !(_DWORD)result )
    {
      sub_18DFB40(a1);
      result = *(unsigned int *)(a1 + 120);
      if ( !(_DWORD)result )
        return result;
    }
    v9 = (unsigned int)result;
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a1 + 120) = result - 1;
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    {
      v11 = *(__int64 **)(v10 - 8);
      v12 = 3LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      v13 = &v11[3 * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)];
    }
    else
    {
      v13 = (__int64 *)v10;
      v12 = 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      v11 = (__int64 *)(v10 - v12);
    }
    for ( i = v11; v13 != i; i += 3 )
    {
      if ( *(_BYTE *)(*i + 16) > 0x17u )
        sub_18DF510(a1, *i, v12, v9, a5, a6);
    }
    if ( *(_BYTE *)(v10 + 16) == 77 )
    {
      v24[0] = *(_QWORD *)(v10 + 40);
      v15 = sub_18DF0D0(v6, v24);
      if ( !*(_BYTE *)(v15 + 2) )
      {
        *(_BYTE *)(v15 + 2) = 1;
        v16 = *(_QWORD *)(*(_QWORD *)(v15 + 16) + 8LL);
        if ( v16 )
        {
          while ( 1 )
          {
            v17 = sub_1648700(v16);
            if ( (unsigned __int8)(*((_BYTE *)v17 + 16) - 25) <= 9u )
              break;
            v16 = *(_QWORD *)(v16 + 8);
            if ( !v16 )
              goto LABEL_15;
          }
LABEL_20:
          v24[0] = v17[5];
          v20 = sub_18DF0D0(v6, v24);
          if ( *(_BYTE *)(v20 + 3) )
            goto LABEL_18;
          *(_BYTE *)(v20 + 3) = 1;
          v18 = v24[0];
          v19 = *(__int64 **)(a1 + 1624);
          if ( *(__int64 **)(a1 + 1632) != v19 )
            goto LABEL_17;
          v21 = &v19[*(unsigned int *)(a1 + 1644)];
          a5 = *(unsigned int *)(a1 + 1644);
          if ( v19 == v21 )
          {
LABEL_33:
            if ( (unsigned int)a5 < *(_DWORD *)(a1 + 1640) )
            {
              a5 = (unsigned int)(a5 + 1);
              *(_DWORD *)(a1 + 1644) = a5;
              *v21 = v18;
              ++*(_QWORD *)(a1 + 1616);
              goto LABEL_18;
            }
LABEL_17:
            sub_16CCBA0(v23, v24[0]);
            goto LABEL_18;
          }
          v22 = 0;
          while ( v24[0] != *v19 )
          {
            if ( *v19 == -2 )
              v22 = v19;
            if ( v21 == ++v19 )
            {
              if ( v22 )
              {
                *v22 = v24[0];
                --*(_DWORD *)(a1 + 1648);
                ++*(_QWORD *)(a1 + 1616);
                break;
              }
              goto LABEL_33;
            }
          }
LABEL_18:
          while ( 1 )
          {
            v16 = *(_QWORD *)(v16 + 8);
            if ( !v16 )
              break;
            v17 = sub_1648700(v16);
            if ( (unsigned __int8)(*((_BYTE *)v17 + 16) - 25) <= 9u )
              goto LABEL_20;
          }
        }
      }
    }
LABEL_15:
    LODWORD(result) = *(_DWORD *)(a1 + 120);
  }
}
