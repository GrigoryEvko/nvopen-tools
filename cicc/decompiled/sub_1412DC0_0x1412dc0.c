// Function: sub_1412DC0
// Address: 0x1412dc0
//
__int64 __fastcall sub_1412DC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int i; // r12d
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r15
  __int64 v13; // r14
  unsigned int v14; // r13d
  unsigned int v15; // eax
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-50h]
  _QWORD v23[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 11 )
  {
    v7 = a3;
    if ( !(unsigned __int8)sub_15F32D0(a4)
      && (*(_BYTE *)(a4 + 18) & 1) == 0
      && !(unsigned __int8)sub_1560180(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 56LL) + 112LL, 45) )
    {
      v9 = sub_15F2050(a4);
      v10 = sub_1632FA0(v9);
      v11 = *(_QWORD *)(a4 - 24);
      v23[0] = 0;
      v12 = v10;
      if ( a1 == sub_14AC610(v11, v23, v10) && v23[0] <= a2 )
      {
        v13 = v7 + a2;
        v22 = v23[0];
        v14 = 1 << (*(unsigned __int16 *)(a4 + 18) >> 1) >> 1;
        if ( v23[0] + v14 >= v7 + a2 )
        {
          v15 = sub_1643030(*(_QWORD *)a4);
          v16 = v22;
          v17 = (v15 >> 3) | ((unsigned __int64)(v15 >> 3) >> 1);
          v18 = (((v17 >> 2) | v17) >> 4) | (v17 >> 2) | v17;
          for ( i = ((((v18 >> 8) | v18) >> 16) | (v18 >> 8) | v18) + 1; i <= v14; i *= 2 )
          {
            v19 = *(unsigned __int8 **)(v12 + 24);
            v20 = &v19[*(unsigned int *)(v12 + 32)];
            if ( v19 == v20 )
              break;
            while ( 8 * i > *v19 )
            {
              if ( v20 == ++v19 )
                return 0;
            }
            v21 = i + v16;
            if ( v21 > v13 )
            {
              if ( (unsigned __int8)sub_1560180(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 56LL) + 112LL, 42)
                || (unsigned __int8)sub_1560180(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 56LL) + 112LL, 43) )
              {
                return 0;
              }
              v16 = v23[0];
              v21 = i + v23[0];
            }
            if ( v13 <= v21 )
              return i;
          }
        }
      }
    }
  }
  return 0;
}
