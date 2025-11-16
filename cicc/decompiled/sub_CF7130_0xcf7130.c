// Function: sub_CF7130
// Address: 0xcf7130
//
__int64 __fastcall sub_CF7130(__int64 a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4, __int64 a5)
{
  int v5; // r15d
  __int64 v6; // rdx
  int v7; // edx
  unsigned __int8 *v8; // r14
  __int64 v9; // rbx
  unsigned __int8 *v10; // rbx
  unsigned int v11; // r15d
  unsigned int v12; // r13d
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-B8h]
  int v23; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-A8h]
  _QWORD v26[6]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v27[12]; // [rsp+60h] [rbp-60h] BYREF

  if ( a4
    && (v5 = a4, v24 = sub_98ACB0(*a3, 6u), (unsigned __int8)sub_CF70D0(v24))
    && (unsigned __int8)(*a2 - 34) <= 0x33u
    && (v6 = 0x8000000000041LL, _bittest64(&v6, (unsigned int)*a2 - 34))
    && v24 != a2
    && !(unsigned __int8)sub_D13FF0((_DWORD)v24, 1, (_DWORD)a2, v5, 1, 0, 0) )
  {
    v7 = *a2;
    v8 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( v7 == 40 )
    {
      v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v9 = -32;
      if ( v7 != 85 )
      {
        v9 = -96;
        if ( v7 != 34 )
LABEL_41:
          BUG();
      }
    }
    v10 = &a2[v9];
    v11 = 0;
    if ( v10 != v8 )
    {
      v12 = 0;
      do
      {
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v8 + 8LL) + 8LL) != 14 )
          goto LABEL_11;
        if ( (unsigned __int8)sub_B49EE0(a2, v12) )
          goto LABEL_11;
        v27[1] = -1;
        v27[0] = v24;
        v13 = *(_QWORD *)v8;
        memset(&v27[2], 0, 32);
        v26[0] = v13;
        v26[1] = -1;
        memset(&v26[2], 0, 32);
        if ( !(unsigned __int8)sub_CF4D50(a1, (__int64)v26, (__int64)v27, a5, (__int64)a2) || sub_CF49B0(a2, v12, 50) )
          goto LABEL_11;
        v14 = *a2;
        if ( v14 == 40 )
        {
          v15 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
        }
        else
        {
          v15 = 0;
          if ( v14 != 85 )
          {
            if ( v14 != 34 )
              goto LABEL_41;
            v15 = 64;
          }
        }
        if ( (a2[7] & 0x80u) != 0 )
        {
          v17 = sub_BD2BC0((__int64)a2);
          v22 = v18 + v17;
          if ( (a2[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)(v22 >> 4) )
LABEL_43:
              BUG();
          }
          else if ( (unsigned int)((v22 - sub_BD2BC0((__int64)a2)) >> 4) )
          {
            if ( (a2[7] & 0x80u) == 0 )
              goto LABEL_43;
            v23 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
            if ( (a2[7] & 0x80u) == 0 )
              BUG();
            v19 = sub_BD2BC0((__int64)a2);
            v21 = 32LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v23);
            goto LABEL_35;
          }
        }
        v21 = 0;
LABEL_35:
        if ( (v12 >= (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v15 - v21) >> 5)
           || !(unsigned __int8)sub_B49B80((__int64)a2, v12, 81))
          && !sub_CF49B0(a2, v12, 51)
          && !sub_CF49B0(a2, v12, 50) )
        {
          return 3;
        }
        v11 = 1;
LABEL_11:
        v8 += 32;
        ++v12;
      }
      while ( v10 != v8 );
    }
  }
  else
  {
    return 3;
  }
  return v11;
}
