// Function: sub_2CE2D50
// Address: 0x2ce2d50
//
__int64 __fastcall sub_2CE2D50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rbx
  unsigned __int64 *v10; // r14
  char v11; // al
  unsigned __int64 v12; // rcx
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // r9
  unsigned __int64 *v15; // rdx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  unsigned __int64 *v18; // r9
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v23; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 *v24[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a2 != a3 )
  {
    v4 = *(_QWORD *)(a2 + 40);
    if ( v4 != *(_QWORD *)(a3 + 40) )
      return 1;
    v6 = a2 + 24;
    v7 = v4 + 48;
    v8 = a3 + 24;
    if ( a2 + 24 == v7 )
      return 1;
    v10 = a1 + 39;
    while ( v8 != v6 )
    {
      if ( !v6 )
        BUG();
      v11 = *(_BYTE *)(v6 - 24);
      if ( v11 == 61 )
      {
        v12 = *(_QWORD *)(v6 - 56);
        v13 = (unsigned __int64 *)a1[40];
        v23 = v12;
        if ( !v13 )
          return 1;
        v14 = v10;
        v15 = v13;
        do
        {
          while ( 1 )
          {
            v16 = v15[2];
            v17 = v15[3];
            if ( v15[4] >= v12 )
              break;
            v15 = (unsigned __int64 *)v15[3];
            if ( !v17 )
              goto LABEL_14;
          }
          v14 = v15;
          v15 = (unsigned __int64 *)v15[2];
        }
        while ( v16 );
LABEL_14:
        if ( v10 == v14 || v14[4] > v12 )
          return 1;
        v18 = v10;
        do
        {
          while ( 1 )
          {
            v19 = v13[2];
            v20 = v13[3];
            if ( v13[4] >= v12 )
              break;
            v13 = (unsigned __int64 *)v13[3];
            if ( !v20 )
              goto LABEL_20;
          }
          v18 = v13;
          v13 = (unsigned __int64 *)v13[2];
        }
        while ( v19 );
LABEL_20:
        if ( v10 == v18 || v18[4] > v12 )
        {
          v24[0] = (unsigned __int64 *)&v23;
          v18 = sub_2CE2AA0(a1 + 38, (__int64)v18, v24);
        }
        if ( v18[5] == a4 )
          return 1;
        v11 = *(_BYTE *)(v6 - 24);
      }
      if ( v11 != 85
        || (v21 = *(_QWORD *)(v6 - 56)) != 0
        && !*(_BYTE *)v21
        && *(_QWORD *)(v21 + 24) == *(_QWORD *)(v6 + 56)
        && (*(_BYTE *)(v21 + 33) & 0x20) != 0
        && (v24[0] = *(unsigned __int64 **)(v6 + 48), v23 = sub_A74680(v24), (unsigned __int8)sub_A73170(&v23, 50)) )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 != v7 )
          continue;
      }
      return 1;
    }
  }
  return 0;
}
