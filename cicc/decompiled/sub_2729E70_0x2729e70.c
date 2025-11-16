// Function: sub_2729E70
// Address: 0x2729e70
//
unsigned __int64 __fastcall sub_2729E70(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned __int64 result; // rax
  int v8; // edx
  unsigned __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  int i; // esi
  unsigned __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 *v21; // rdx
  unsigned int v22; // [rsp+4h] [rbp-5Ch]
  unsigned __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  int v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v28; // [rsp+20h] [rbp-40h]

  v4 = a2 + 48;
  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == result )
    goto LABEL_42;
  if ( !result )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(result - 24) - 30 > 0xA )
LABEL_42:
    BUG();
  if ( *(_BYTE *)(result - 24) != 31 )
    return result;
  if ( (*(_DWORD *)(result - 20) & 0x7FFFFFF) != 3 )
    return result;
  result = *(_QWORD *)(result - 120);
  v24 = result;
  if ( *(_BYTE *)result != 82 )
    return result;
  result = *(_QWORD *)(result - 32);
  if ( *(_BYTE *)result > 0x15u )
    return result;
  v22 = sub_B53900(v24);
  result = v22 - 32;
  if ( (unsigned int)result > 1 )
    return result;
  v8 = *a1;
  v23 = *(_QWORD *)(v24 - 64);
  v9 = (unsigned __int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( v8 == 40 )
  {
    v27 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v27 = -32;
    if ( v8 != 85 )
    {
      v27 = -96;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v10 = sub_BD2BC0((__int64)a1);
    v25 = v11 + v10;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v25 >> 4) )
        goto LABEL_18;
    }
    else
    {
      if ( !(unsigned int)((v25 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_18;
      if ( (a1[7] & 0x80u) != 0 )
      {
        v26 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v12 = sub_BD2BC0((__int64)a1);
        v27 -= 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v26);
        goto LABEL_18;
      }
    }
    BUG();
  }
LABEL_18:
  result = (unsigned __int64)&a1[v27];
  v28 = (unsigned __int64 *)result;
  if ( v9 != (unsigned __int64 *)result )
  {
    for ( i = 0; ; ++i )
    {
      result = *v9;
      if ( *(_BYTE *)*v9 > 0x15u )
      {
        result = sub_B49B80((__int64)a1, i, 43);
        if ( !(_BYTE)result )
        {
          result = v23;
          if ( v23 == *v9 )
            break;
        }
      }
      v9 += 4;
      if ( v9 == v28 )
        return result;
    }
    v15 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == v15 )
    {
      v17 = 0;
    }
    else
    {
      if ( !v15 )
        BUG();
      v16 = *(unsigned __int8 *)(v15 - 24);
      v17 = 0;
      v18 = v15 - 24;
      if ( (unsigned int)(v16 - 30) < 0xB )
        v17 = v18;
    }
    if ( a3 != sub_B46EC0(v17, 0) )
      v22 = sub_B52870(*(_WORD *)(v24 + 2) & 0x3F);
    v20 = *(unsigned int *)(a4 + 8);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v20 + 1, 0x10u, v20 + 1, v19);
      v20 = *(unsigned int *)(a4 + 8);
    }
    v21 = (__int64 *)(*(_QWORD *)a4 + 16 * v20);
    v21[1] = v22;
    *v21 = v24;
    ++*(_DWORD *)(a4 + 8);
    return v24;
  }
  return result;
}
