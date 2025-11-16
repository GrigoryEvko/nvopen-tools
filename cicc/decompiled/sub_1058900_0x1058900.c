// Function: sub_1058900
// Address: 0x1058900
//
__int64 __fastcall sub_1058900(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 i; // r12
  unsigned __int8 *v8; // r15
  char v9; // al
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned __int8 *v15; // rbx
  __int64 result; // rax
  unsigned __int8 *v17; // r12
  __int64 v18; // rsi
  char v19; // al
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v3 = a1;
  v4 = *(_QWORD *)(a1 + 216);
  v5 = *(_QWORD *)(v4 + 80);
  v6 = v4 + 72;
  if ( v4 + 72 == v5 )
  {
    i = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v5 + 32);
      if ( i != v5 + 24 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        break;
      if ( !v5 )
        BUG();
    }
  }
LABEL_7:
  while ( v6 != v5 )
  {
    v8 = (unsigned __int8 *)(i - 24);
    v25 = v3;
    if ( !i )
      v8 = 0;
    a2 = (__int64)v8;
    v9 = sub_DF9740(*(__int64 **)(v3 + 232), v8);
    v3 = v25;
    if ( v9 )
    {
      if ( *(_BYTE *)(v25 + 1284) )
      {
        v12 = *(__int64 **)(v25 + 1264);
        a3 = &v12[*(unsigned int *)(v25 + 1276)];
        if ( v12 != a3 )
        {
          while ( v8 != (unsigned __int8 *)*v12 )
          {
            if ( a3 == ++v12 )
              goto LABEL_40;
          }
          goto LABEL_16;
        }
      }
      else
      {
        a2 = (__int64)v8;
        v23 = sub_C8CA60(v25 + 1256, (__int64)v8);
        v3 = v25;
        if ( v23 )
          goto LABEL_16;
      }
LABEL_40:
      a2 = (__int64)v8;
      v27 = v3;
      sub_1057F60(v3, v8, a3, v3, v10, v11);
      v3 = v27;
    }
    else
    {
      a2 = (__int64)v8;
      v19 = sub_DF97C0(*(_QWORD *)(v25 + 232));
      v3 = v25;
      if ( v19 )
      {
        if ( !*(_BYTE *)(v25 + 1284) )
          goto LABEL_42;
        v22 = *(__int64 **)(v25 + 1264);
        a2 = *(unsigned int *)(v25 + 1276);
        a3 = &v22[a2];
        if ( v22 != a3 )
        {
          while ( v8 != (unsigned __int8 *)*v22 )
          {
            if ( a3 == ++v22 )
              goto LABEL_41;
          }
          goto LABEL_16;
        }
LABEL_41:
        if ( (unsigned int)a2 < *(_DWORD *)(v25 + 1272) )
        {
          a2 = (unsigned int)(a2 + 1);
          *(_DWORD *)(v25 + 1276) = a2;
          *a3 = (__int64)v8;
          ++*(_QWORD *)(v25 + 1256);
        }
        else
        {
LABEL_42:
          a2 = (__int64)v8;
          sub_C8CC70(v25 + 1256, (__int64)v8, (__int64)a3, v25, v20, v21);
          v3 = v25;
        }
      }
    }
LABEL_16:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v5 + 32) )
    {
      v13 = v5 - 24;
      if ( !v5 )
        v13 = 0;
      if ( i != v13 + 48 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        goto LABEL_7;
      if ( !v5 )
        BUG();
    }
  }
  v14 = *(_QWORD *)(v3 + 216);
  if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
  {
    v28 = v3;
    sub_B2C6D0(*(_QWORD *)(v3 + 216), a2, (__int64)a3, v3);
    v15 = *(unsigned __int8 **)(v14 + 96);
    v3 = v28;
    result = 5LL * *(_QWORD *)(v14 + 104);
    v17 = &v15[40 * *(_QWORD *)(v14 + 104)];
    if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
    {
      result = sub_B2C6D0(v14, a2, v24, v28);
      v15 = *(unsigned __int8 **)(v14 + 96);
      v3 = v28;
    }
  }
  else
  {
    v15 = *(unsigned __int8 **)(v14 + 96);
    result = 5LL * *(_QWORD *)(v14 + 104);
    v17 = &v15[40 * *(_QWORD *)(v14 + 104)];
  }
  for ( ; v17 != v15; v3 = v26 )
  {
    while ( 1 )
    {
      v26 = v3;
      result = sub_DF9740(*(__int64 **)(v3 + 232), v15);
      v3 = v26;
      if ( (_BYTE)result )
        break;
      v15 += 40;
      if ( v17 == v15 )
        return result;
    }
    v18 = (__int64)v15;
    v15 += 40;
    result = sub_1057CE0(v26, v18);
  }
  return result;
}
