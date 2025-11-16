// Function: sub_2CE2B60
// Address: 0x2ce2b60
//
__int64 __fastcall sub_2CE2B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // r13
  __int64 v6; // r15
  unsigned __int64 *v8; // rbx
  _QWORD *v9; // r9
  unsigned __int64 v10; // rcx
  _QWORD *v11; // rdx
  unsigned __int64 *v12; // r10
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // r10
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned __int64 *v20; // rax
  unsigned __int64 *v21; // rsi
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a2 + 8;
  v6 = *(_QWORD *)(a2 + 24);
  if ( v6 != a2 + 8 )
  {
    v25 = 0;
    v8 = a5 + 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = (_QWORD *)a5[2];
        v10 = *(_QWORD *)(*(_QWORD *)(v6 + 32) - 32LL);
        v11 = v9;
        v27 = v10;
        if ( !v9 )
          goto LABEL_17;
        v12 = v8;
        v13 = v9;
        do
        {
          while ( 1 )
          {
            v14 = v13[2];
            v15 = v13[3];
            if ( v13[4] >= v10 )
              break;
            v13 = (unsigned __int64 *)v13[3];
            if ( !v15 )
              goto LABEL_8;
          }
          v12 = v13;
          v13 = (unsigned __int64 *)v13[2];
        }
        while ( v14 );
LABEL_8:
        if ( v12 == v8 || v12[4] > v10 )
          goto LABEL_17;
        v16 = (__int64)v8;
        do
        {
          while ( 1 )
          {
            v17 = v11[2];
            v18 = v11[3];
            if ( v11[4] >= v10 )
              break;
            v11 = (_QWORD *)v11[3];
            if ( !v18 )
              goto LABEL_14;
          }
          v16 = (__int64)v11;
          v11 = (_QWORD *)v11[2];
        }
        while ( v17 );
LABEL_14:
        if ( (unsigned __int64 *)v16 != v8 && *(_QWORD *)(v16 + 32) <= v10 )
          break;
        v28[0] = &v27;
        if ( sub_2CE2AA0(a5, v16, v28)[5] != a3 )
          goto LABEL_17;
        v20 = (unsigned __int64 *)a5[2];
        if ( !v20 )
        {
          v21 = v8;
LABEL_26:
          v28[0] = &v27;
          v21 = sub_2CE2AA0(a5, (__int64)v21, v28);
          goto LABEL_27;
        }
LABEL_20:
        v21 = v8;
        do
        {
          while ( 1 )
          {
            v22 = v20[2];
            v23 = v20[3];
            if ( v20[4] >= v27 )
              break;
            v20 = (unsigned __int64 *)v20[3];
            if ( !v23 )
              goto LABEL_24;
          }
          v21 = v20;
          v20 = (unsigned __int64 *)v20[2];
        }
        while ( v22 );
LABEL_24:
        if ( v21 == v8 || v21[4] > v27 )
          goto LABEL_26;
LABEL_27:
        if ( v21[6] != a4 )
          goto LABEL_17;
        if ( v25 )
          return 0;
        v25 = *(_QWORD *)(v6 + 32);
        v6 = sub_220EF30(v6);
        if ( v5 == v6 )
          return v25;
      }
      if ( *(_QWORD *)(v16 + 40) == a3 )
      {
        v20 = v9;
        goto LABEL_20;
      }
LABEL_17:
      v6 = sub_220EF30(v6);
      if ( v5 == v6 )
        return v25;
    }
  }
  return 0;
}
