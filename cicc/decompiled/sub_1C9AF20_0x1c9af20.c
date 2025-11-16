// Function: sub_1C9AF20
// Address: 0x1c9af20
//
__int64 __fastcall sub_1C9AF20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 *v11; // r14
  char v12; // al
  unsigned __int64 v13; // rcx
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // r10
  unsigned __int64 *v16; // rdx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  unsigned __int64 *v19; // r10
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 *v26[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a2 != a3 )
  {
    v4 = *(_QWORD *)(a2 + 40);
    if ( v4 != *(_QWORD *)(a3 + 40) )
      return 1;
    v6 = *(_QWORD *)(v4 + 56);
    v7 = a2 + 24;
    v8 = v4 + 40;
    v9 = a3 + 24;
    v23 = **(__int64 ***)(v6 + 40);
    if ( a2 + 24 == v8 )
      return 1;
    v11 = a1 + 39;
    while ( v9 != v7 )
    {
      if ( !v7 )
        BUG();
      v12 = *(_BYTE *)(v7 - 8);
      if ( v12 == 54 )
      {
        v13 = *(_QWORD *)(v7 - 48);
        v14 = (unsigned __int64 *)a1[40];
        v25 = v13;
        if ( !v14 )
          return 1;
        v15 = v11;
        v16 = v14;
        do
        {
          while ( 1 )
          {
            v17 = v16[2];
            v18 = v16[3];
            if ( v16[4] >= v13 )
              break;
            v16 = (unsigned __int64 *)v16[3];
            if ( !v18 )
              goto LABEL_14;
          }
          v15 = v16;
          v16 = (unsigned __int64 *)v16[2];
        }
        while ( v17 );
LABEL_14:
        if ( v11 == v15 || v15[4] > v13 )
          return 1;
        v19 = v11;
        do
        {
          while ( 1 )
          {
            v20 = v14[2];
            v21 = v14[3];
            if ( v14[4] >= v13 )
              break;
            v14 = (unsigned __int64 *)v14[3];
            if ( !v21 )
              goto LABEL_20;
          }
          v19 = v14;
          v14 = (unsigned __int64 *)v14[2];
        }
        while ( v20 );
LABEL_20:
        if ( v11 == v19 || v19[4] > v13 )
        {
          v26[0] = (unsigned __int64 *)&v25;
          v19 = sub_1C9AC70(a1 + 38, v19, v26);
        }
        if ( v19[5] == a4 )
          return 1;
        v12 = *(_BYTE *)(v7 - 8);
      }
      if ( v12 != 78
        || (v22 = *(_QWORD *)(v7 - 48), !*(_BYTE *)(v22 + 16))
        && (*(_BYTE *)(v22 + 33) & 0x20) != 0
        && (v26[0] = (unsigned __int64 *)sub_15E1850(v23, *(_DWORD *)(v22 + 36)),
            v25 = sub_1560250(v26),
            (unsigned __int8)sub_155EE10((__int64)&v25, 36)) )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v7 != v8 )
          continue;
      }
      return 1;
    }
  }
  return 0;
}
