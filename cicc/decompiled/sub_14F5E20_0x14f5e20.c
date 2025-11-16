// Function: sub_14F5E20
// Address: 0x14f5e20
//
void __fastcall sub_14F5E20(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rcx
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  _QWORD *v8; // r8
  _QWORD *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rdx
  _QWORD *v14; // r14
  _QWORD *v15; // r8
  _QWORD *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // r8
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r8
  _QWORD v23[2]; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 *v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v23[0] = a3;
  if ( a3 != a2 )
  {
    v3 = a3;
    if ( a3 )
    {
      v4 = (_QWORD *)a1[18];
      v5 = a1 + 17;
      v8 = a1 + 17;
      if ( !v4 )
      {
LABEL_16:
        v14 = a1 + 16;
LABEL_17:
        v24[0] = v23;
        v8 = (_QWORD *)sub_14F5AE0(v14, v8, v24);
LABEL_18:
        v8[5] = a2;
        return;
      }
      v9 = (_QWORD *)a1[18];
      do
      {
        while ( 1 )
        {
          v10 = v9[2];
          v11 = v9[3];
          if ( v3 <= v9[4] )
            break;
          v9 = (_QWORD *)v9[3];
          if ( !v11 )
            goto LABEL_8;
        }
        v8 = v9;
        v9 = (_QWORD *)v9[2];
      }
      while ( v10 );
LABEL_8:
      if ( v5 == v8 || v3 < v8[4] )
      {
LABEL_10:
        v8 = v5;
        do
        {
          while ( 1 )
          {
            v12 = v4[2];
            v13 = v4[3];
            if ( v4[4] >= v3 )
              break;
            v4 = (_QWORD *)v4[3];
            if ( !v13 )
              goto LABEL_14;
          }
          v8 = v4;
          v4 = (_QWORD *)v4[2];
        }
        while ( v12 );
LABEL_14:
        if ( v5 != v8 && v3 >= v8[4] )
          goto LABEL_18;
        goto LABEL_16;
      }
      v15 = v5;
      v16 = v4;
      do
      {
        while ( 1 )
        {
          v17 = v16[2];
          v18 = v16[3];
          if ( v3 <= v16[4] )
            break;
          v16 = (_QWORD *)v16[3];
          if ( !v18 )
            goto LABEL_24;
        }
        v15 = v16;
        v16 = (_QWORD *)v16[2];
      }
      while ( v17 );
LABEL_24:
      if ( v5 == v15 || v3 < v15[4] )
      {
        v14 = a1 + 16;
        v24[0] = v23;
        v22 = sub_14F5AE0(a1 + 16, v15, v24);
        v4 = (_QWORD *)a1[18];
        if ( *(_QWORD *)(v22 + 40) == a2 )
        {
          if ( !v4 )
          {
            v8 = v5;
            goto LABEL_17;
          }
          v3 = v23[0];
          goto LABEL_10;
        }
        if ( !v4 )
        {
          v19 = v5;
          goto LABEL_34;
        }
        v3 = v23[0];
      }
      else if ( a2 == v15[5] )
      {
        goto LABEL_10;
      }
      v19 = v5;
      do
      {
        while ( 1 )
        {
          v20 = v4[2];
          v21 = v4[3];
          if ( v4[4] >= v3 )
            break;
          v4 = (_QWORD *)v4[3];
          if ( !v21 )
            goto LABEL_31;
        }
        v19 = v4;
        v4 = (_QWORD *)v4[2];
      }
      while ( v20 );
LABEL_31:
      if ( v5 != v19 && v3 >= v19[4] )
        goto LABEL_35;
      v14 = a1 + 16;
LABEL_34:
      v24[0] = v23;
      v19 = (_QWORD *)sub_14F5AE0(v14, v19, v24);
LABEL_35:
      v19[5] = 0;
    }
  }
}
