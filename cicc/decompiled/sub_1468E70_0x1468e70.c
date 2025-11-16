// Function: sub_1468E70
// Address: 0x1468e70
//
__int64 __fastcall sub_1468E70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned int v6; // r15d
  __int64 v7; // r14
  __int64 v8; // rbx
  unsigned int v9; // eax
  unsigned int v11; // [rsp+Ch] [rbp-54h]
  __int64 v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-38h]

  v11 = *(_DWORD *)(a3 + 32);
  v13 = v11;
  if ( v11 > 0x40 )
  {
    sub_16A4FD0(&v12, a3 + 24);
    v11 = v13;
  }
  else
  {
    v12 = *(_QWORD *)(a3 + 24);
  }
  v5 = *(_QWORD *)(a4 + 40);
  if ( (unsigned int)v5 <= 1 )
  {
    if ( v11 )
    {
LABEL_18:
      *(_DWORD *)(a1 + 8) = v11;
      if ( v11 > 0x40 )
      {
        sub_16A4FD0(a1, &v12);
        v11 = v13;
      }
      else
      {
        *(_QWORD *)a1 = v12;
      }
    }
    else
    {
      *(_DWORD *)(a1 + 8) = 0;
LABEL_10:
      *(_QWORD *)a1 = 0;
      v11 = v13;
    }
  }
  else
  {
    v6 = v11;
    v7 = 8;
    v8 = 8LL * (unsigned int)v5;
    do
    {
      if ( !v6 )
        goto LABEL_9;
      v9 = sub_14687F0(a2, *(_QWORD *)(*(_QWORD *)(a4 + 32) + v7));
      if ( v6 > v9 )
        v6 = v9;
      v7 += 8;
    }
    while ( v8 != v7 );
    if ( !v6 )
    {
LABEL_9:
      *(_DWORD *)(a1 + 8) = v11;
      if ( v11 <= 0x40 )
        goto LABEL_10;
      sub_16A4EF0(a1, 0, 0);
      v11 = v13;
      goto LABEL_20;
    }
    if ( v6 >= v11 )
    {
      v11 = v13;
      goto LABEL_18;
    }
    sub_16A5A50(&v14, &v12);
    sub_16A5C50(a1, &v14, v11);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    v11 = v13;
  }
LABEL_20:
  if ( v11 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return a1;
}
