// Function: sub_10E8D80
// Address: 0x10e8d80
//
__int64 __fastcall sub_10E8D80(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  unsigned int v5; // r14d
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  int v12; // edx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // [rsp+0h] [rbp-50h]
  int v24; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-40h]

  v3 = a1 + 24;
  v26 = *((_QWORD *)a1 + 5) + 48LL;
  if ( a1 + 24 != (unsigned __int8 *)v26 )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)v3 - 24) != 85 )
        return 0;
      v7 = *(v3 - 7);
      if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(v7 + 24) != v3[7] || (*(_BYTE *)(v7 + 33) & 0x20) == 0 )
        return 0;
      if ( !sub_B46AA0((__int64)(v3 - 3)) )
      {
        v9 = *((_QWORD *)a1 - 4);
        if ( !v9 || *(_BYTE *)v9 || (v10 = *((_QWORD *)a1 + 10), *(_QWORD *)(v9 + 24) != v10) )
          BUG();
        v11 = *(unsigned int *)(v7 + 36);
        if ( *(_DWORD *)(v9 + 36) != (_DWORD)v11 )
          break;
      }
LABEL_30:
      v22 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      v3 = (_QWORD *)v22;
      if ( v26 == v22 )
        return 0;
      if ( !v22 )
        BUG();
    }
    if ( !*(_QWORD *)(a3 + 16) )
      sub_4263D6(v11, v10, v8);
    v5 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(a3 + 24))(a3, v3 - 3);
    if ( !(_BYTE)v5 )
      return 0;
    v12 = *a1;
    if ( v12 == 40 )
    {
      v13 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
    }
    else
    {
      v13 = 0;
      if ( v12 != 85 )
      {
        if ( v12 != 34 )
          goto LABEL_41;
        v13 = 64;
      }
    }
    if ( (a1[7] & 0x80u) != 0 )
    {
      v14 = sub_BD2BC0((__int64)a1);
      v23 = v15 + v14;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v23 >> 4) )
LABEL_41:
          BUG();
      }
      else if ( (unsigned int)((v23 - sub_BD2BC0((__int64)a1)) >> 4) )
      {
        if ( (a1[7] & 0x80u) == 0 )
          goto LABEL_41;
        v24 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v16 = sub_BD2BC0((__int64)a1);
        v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v24);
        goto LABEL_26;
      }
    }
    v18 = 0;
LABEL_26:
    v19 = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v13 - v18) >> 5;
    if ( !(_DWORD)v19 )
    {
LABEL_35:
      sub_F207A0(a2, v3 - 3);
      sub_F207A0(a2, (__int64 *)a1);
      return v5;
    }
    v20 = 32LL * (unsigned int)v19;
    v21 = 0;
    while ( *(_QWORD *)&a1[v21 + -32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)] == v3[v21 / 8
                                                                              - 3
                                                                              + -4 * (*((_DWORD *)v3 - 5) & 0x7FFFFFF)] )
    {
      v21 += 32LL;
      if ( v20 == v21 )
        goto LABEL_35;
    }
    goto LABEL_30;
  }
  return 0;
}
