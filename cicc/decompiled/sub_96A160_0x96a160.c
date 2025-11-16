// Function: sub_96A160
// Address: 0x96a160
//
__int64 __fastcall sub_96A160(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 v5; // r13
  char v6; // al
  _QWORD *k; // rbx
  _QWORD *i; // r13
  _QWORD *j; // r13
  int v12; // [rsp+1Ch] [rbp-94h] BYREF
  __int64 v13; // [rsp+20h] [rbp-90h] BYREF
  __int64 v14; // [rsp+28h] [rbp-88h]
  char v15; // [rsp+34h] [rbp-7Ch]
  _BYTE v16[8]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v17; // [rsp+48h] [rbp-68h]
  _BYTE v18[8]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD *v19; // [rsp+68h] [rbp-48h]

  v2 = *(_QWORD *)(a1 + 24);
  v3 = a1 + 24;
  v4 = sub_C33340();
  if ( v2 == v4 )
  {
    sub_C41050(v16, v3, &v12, 1);
    sub_C3C840(v18, v16);
    sub_C3C840(&v13, v18);
    if ( v19 )
    {
      for ( i = &v19[3 * *(v19 - 1)]; v19 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v4 == *i )
            break;
          sub_C338F0(i);
          if ( v19 == i )
            goto LABEL_16;
        }
      }
LABEL_16:
      j_j_j___libc_free_0_0(i - 1);
    }
    if ( v17 )
    {
      for ( j = &v17[3 * *(v17 - 1)]; v17 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v4 == *j )
            break;
          sub_C338F0(j);
          if ( v17 == j )
            goto LABEL_19;
        }
      }
LABEL_19:
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C3C390(v16, v3, &v12, 1);
    sub_C338E0(v18, v16);
    sub_C407B0(&v13, v18, v2);
    sub_C338F0(v18);
    sub_C338F0(v16);
  }
  v5 = sub_AD8F10(*(_QWORD *)(a1 + 8), &v13);
  if ( v4 == v13 )
  {
    v6 = *(_BYTE *)(v14 + 20) & 7;
    if ( v6 == 1 )
      goto LABEL_5;
  }
  else
  {
    v6 = v15 & 7;
    if ( (v15 & 7) == 1 )
      goto LABEL_5;
  }
  if ( v6 )
  {
    sub_AD64C0(a2, v12, 1);
    if ( v4 != v13 )
      goto LABEL_6;
    goto LABEL_11;
  }
LABEL_5:
  sub_AD6530(a2);
  if ( v4 != v13 )
  {
LABEL_6:
    sub_C338F0(&v13);
    return v5;
  }
LABEL_11:
  if ( v14 )
  {
    for ( k = (_QWORD *)(v14 + 24LL * *(_QWORD *)(v14 - 8)); (_QWORD *)v14 != k; sub_969EE0((__int64)k) )
    {
      while ( 1 )
      {
        k -= 3;
        if ( v4 == *k )
          break;
        sub_C338F0(k);
        if ( (_QWORD *)v14 == k )
          goto LABEL_13;
      }
    }
LABEL_13:
    j_j_j___libc_free_0_0(k - 1);
  }
  return v5;
}
