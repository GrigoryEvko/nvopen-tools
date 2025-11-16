// Function: sub_1AC0A70
// Address: 0x1ac0a70
//
__int64 __fastcall sub_1AC0A70(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r15
  int v6; // ebx
  int v7; // r14d
  __int64 v8; // rcx
  int v9; // edi
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r15
  int v16; // r8d
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  int v19; // ebx
  int v20; // r14d
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rdx
  _QWORD *v27; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v28[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v29; // [rsp+30h] [rbp-40h]

  v2 = a2[1];
  v27 = a2;
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = sub_1648700(v2);
      if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 25) <= 9u )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        goto LABEL_18;
    }
    v4 = 0;
    v5 = *(_QWORD *)(a1 + 48);
    v6 = *(_DWORD *)(a1 + 64);
    v7 = v6 - 1;
LABEL_7:
    if ( !v6 )
      goto LABEL_5;
    v8 = v3[5];
    v9 = 1;
    v10 = v7 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v11 = *(_QWORD *)(v5 + 8LL * v10);
    if ( v11 != v8 )
    {
      while ( v11 != -8 )
      {
        v10 = v7 & (v9 + v10);
        v11 = *(_QWORD *)(v5 + 8LL * v10);
        if ( v8 == v11 )
          goto LABEL_9;
        ++v9;
      }
LABEL_5:
      while ( 1 )
      {
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          break;
LABEL_6:
        v3 = sub_1648700(v2);
        if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 25) <= 9u )
          goto LABEL_7;
      }
LABEL_11:
      if ( v4 )
        return v4;
      goto LABEL_18;
    }
LABEL_9:
    if ( !v4 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      v4 = v8;
      if ( v2 )
        goto LABEL_6;
      goto LABEL_11;
    }
    if ( v8 == v4 )
      goto LABEL_5;
  }
LABEL_18:
  v29 = 257;
  v13 = sub_157ED20((__int64)a2);
  v14 = sub_157FBF0(a2, (__int64 *)(v13 + 24), (__int64)v28);
  v15 = v27[1];
  if ( v15 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v15) + 16) - 25) > 9u )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        goto LABEL_21;
    }
    do
    {
      v18 = v15;
      v19 = *(_DWORD *)(a1 + 64);
      v20 = v19 - 1;
      do
      {
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          break;
LABEL_27:
        ;
      }
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v15) + 16) - 25) > 9u );
      while ( 1 )
      {
        v21 = sub_1648700(v18)[5];
        if ( !v19 )
          break;
        v22 = *(_QWORD *)(a1 + 48);
        v23 = v20 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v24 = *(_QWORD *)(v22 + 8LL * v23);
        if ( v24 != v21 )
        {
          v16 = 1;
          while ( v24 != -8 )
          {
            v23 = v20 & (v16 + v23);
            v24 = *(_QWORD *)(v22 + 8LL * v23);
            if ( v21 == v24 )
              goto LABEL_30;
            ++v16;
          }
          break;
        }
LABEL_30:
        if ( !v15 )
          goto LABEL_21;
        v18 = v15;
        v15 = *(_QWORD *)(v15 + 8);
        if ( v15 )
          goto LABEL_27;
      }
      v17 = sub_157EBA0(v21);
      sub_1648780(v17, (__int64)v27, v14);
    }
    while ( v15 );
  }
LABEL_21:
  sub_1ABFB80(a1 + 40, (__int64 *)&v27);
  return (__int64)v27;
}
