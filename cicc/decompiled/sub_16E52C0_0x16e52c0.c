// Function: sub_16E52C0
// Address: 0x16e52c0
//
__int64 __fastcall sub_16E52C0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // r14
  size_t v6; // rdx
  const char *v7; // rsi
  const char *v8; // r13
  __int64 v9; // r10
  __int64 v10; // rbx
  unsigned int v11; // esi
  size_t v12; // rdx
  size_t v13; // rdx
  size_t v15; // rdx
  char *v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+18h] [rbp-58h]
  const char *v18[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF

  sub_16E4E00(a1);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = 2;
  v7 = "''";
  if ( v5 )
  {
    if ( !a3 )
    {
      v7 = *(const char **)a2;
      v6 = *(_QWORD *)(a2 + 8);
      return sub_16E4D00(a1, v7, v6);
    }
    v8 = *(const char **)a2;
    if ( a3 == 1 )
    {
      sub_16E4B40(a1, "'", 1u);
      v16 = "'";
    }
    else
    {
      sub_16E4B40(a1, "\"", 1u);
      v16 = "\"";
      if ( a3 == 2 )
      {
        v15 = 0;
        if ( v8 )
          v15 = strlen(v8);
        sub_16F6C30(v18, v8, v15, 0);
        sub_16E4B40(a1, v18[0], (size_t)v18[1]);
        if ( (__int64 *)v18[0] != &v19 )
          j_j___libc_free_0(v18[0], v19 + 1);
        v7 = "\"";
LABEL_12:
        v6 = 1;
        return sub_16E4D00(a1, v7, v6);
      }
    }
    if ( (_DWORD)v5 )
    {
      v9 = (unsigned int)v5;
      v10 = 0;
      v11 = 0;
      do
      {
        while ( *(_BYTE *)(*(_QWORD *)a2 + v10) != 39 )
        {
          if ( v9 == ++v10 )
            goto LABEL_10;
        }
        v12 = (unsigned int)v10 - v11;
        v17 = v9;
        ++v10;
        sub_16E4B40(a1, &v8[v11], v12);
        sub_16E4B40(a1, "''", 2u);
        v9 = v17;
        v11 = v10;
      }
      while ( v17 != v10 );
LABEL_10:
      v8 += v11;
      v13 = (unsigned int)v5 - v11;
    }
    else
    {
      v13 = 0;
    }
    sub_16E4B40(a1, v8, v13);
    v7 = v16;
    goto LABEL_12;
  }
  return sub_16E4D00(a1, v7, v6);
}
