// Function: sub_D46F00
// Address: 0xd46f00
//
__int64 __fastcall sub_D46F00(__int64 a1)
{
  __int64 v1; // r15
  unsigned __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v4; // r12d
  unsigned int v5; // ebx
  __int64 v6; // rsi
  _QWORD *v7; // rdx
  _QWORD *v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdx
  _QWORD *v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // r8
  _QWORD *v17; // rcx
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // rcx
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 *v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  int v38; // [rsp+24h] [rbp-5Ch]
  __int64 *v39; // [rsp+28h] [rbp-58h]
  _QWORD *v40; // [rsp+30h] [rbp-50h]
  _QWORD *v41; // [rsp+30h] [rbp-50h]
  _QWORD *v42; // [rsp+30h] [rbp-50h]
  _QWORD *v43; // [rsp+38h] [rbp-48h]
  _QWORD *v44; // [rsp+38h] [rbp-48h]
  _QWORD *v45; // [rsp+38h] [rbp-48h]
  int v46; // [rsp+40h] [rbp-40h]
  unsigned int v47; // [rsp+44h] [rbp-3Ch]
  unsigned int v48; // [rsp+48h] [rbp-38h]

  v1 = a1 + 56;
  v37 = 0;
  v35 = *(__int64 **)(a1 + 40);
  v39 = *(__int64 **)(a1 + 32);
  if ( v39 == v35 )
    return 0;
  do
  {
    while ( 1 )
    {
      v36 = *v39;
      v2 = *(_QWORD *)(*v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v2 == *v39 + 48 )
        goto LABEL_41;
      if ( !v2 )
        BUG();
      v3 = v2 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
        goto LABEL_41;
      v38 = sub_B46E30(v3);
      v46 = v38 >> 2;
      if ( v38 >> 2 > 0 )
        break;
      v24 = v38;
      v5 = 0;
LABEL_23:
      if ( v24 == 2 )
        goto LABEL_24;
      if ( v24 == 3 )
      {
        v29 = sub_B46EC0(v3, v5);
        if ( !(unsigned __int8)sub_B19060(v1, v29, v30, v31) )
          goto LABEL_25;
        ++v5;
LABEL_24:
        v25 = sub_B46EC0(v3, v5);
        if ( !(unsigned __int8)sub_B19060(v1, v25, v26, v27) )
          goto LABEL_25;
        ++v5;
        goto LABEL_52;
      }
      if ( v24 != 1 )
        goto LABEL_41;
LABEL_52:
      v32 = sub_B46EC0(v3, v5);
      if ( (unsigned __int8)sub_B19060(v1, v32, v33, v34) )
      {
        if ( v35 == ++v39 )
          return v37;
      }
      else
      {
LABEL_25:
        if ( v5 == v38 )
          goto LABEL_41;
LABEL_26:
        if ( v37 )
          return 0;
        ++v39;
        v37 = v36;
        if ( v35 == v39 )
          return v37;
      }
    }
    v48 = 1;
    v4 = 2;
    v47 = 3;
    while ( 1 )
    {
      v5 = v4 - 2;
      v6 = sub_B46EC0(v3, v4 - 2);
      if ( *(_BYTE *)(a1 + 84) )
      {
        v7 = *(_QWORD **)(a1 + 64);
        v8 = &v7[*(unsigned int *)(a1 + 76)];
        if ( v7 != v8 )
        {
          v9 = *(_QWORD **)(a1 + 64);
          do
          {
            if ( v6 == *v9 )
            {
              v40 = &v7[*(unsigned int *)(a1 + 76)];
              v43 = *(_QWORD **)(a1 + 64);
              v5 = v48;
              v10 = sub_B46EC0(v3, v48);
              v11 = v43;
              v12 = v40;
              v13 = v10;
              v14 = v43;
              goto LABEL_14;
            }
            ++v9;
          }
          while ( v8 != v9 );
        }
        goto LABEL_25;
      }
      if ( !sub_C8CA60(v1, v6) )
        goto LABEL_25;
      v5 = v48;
      v13 = sub_B46EC0(v3, v48);
      if ( *(_BYTE *)(a1 + 84) )
      {
        v11 = *(_QWORD **)(a1 + 64);
        v12 = &v11[*(unsigned int *)(a1 + 76)];
        if ( v11 == v12 )
          goto LABEL_25;
        v14 = *(_QWORD **)(a1 + 64);
LABEL_14:
        while ( v13 != *v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_25;
        }
        v41 = v12;
        v5 = v4;
        v44 = v14;
        v15 = sub_B46EC0(v3, v4);
        v16 = v44;
        v17 = v41;
        v18 = v15;
        v19 = v44;
        do
        {
LABEL_17:
          if ( v18 == *v16 )
          {
            v42 = v19;
            v45 = v17;
            v5 = v47;
            v20 = sub_B46EC0(v3, v47);
            v21 = v42;
            v22 = v45;
            v23 = v20;
            goto LABEL_20;
          }
          ++v16;
        }
        while ( v17 != v16 );
        goto LABEL_25;
      }
      if ( !sub_C8CA60(v1, v13) )
        goto LABEL_25;
      v5 = v4;
      v18 = sub_B46EC0(v3, v4);
      if ( *(_BYTE *)(a1 + 84) )
      {
        v16 = *(_QWORD **)(a1 + 64);
        v17 = &v16[*(unsigned int *)(a1 + 76)];
        if ( v17 == v16 )
          goto LABEL_25;
        v19 = *(_QWORD **)(a1 + 64);
        goto LABEL_17;
      }
      if ( !sub_C8CA60(v1, v18) )
        goto LABEL_25;
      v5 = v47;
      v23 = sub_B46EC0(v3, v47);
      if ( !*(_BYTE *)(a1 + 84) )
      {
        if ( !sub_C8CA60(v1, v23) )
          goto LABEL_25;
        goto LABEL_21;
      }
      v21 = *(_QWORD **)(a1 + 64);
      v22 = &v21[*(unsigned int *)(a1 + 76)];
      if ( v21 == v22 )
        break;
LABEL_20:
      while ( v23 != *v21 )
      {
        if ( v22 == ++v21 )
          goto LABEL_25;
      }
LABEL_21:
      v47 += 4;
      v5 = v4 + 2;
      v4 += 4;
      v48 += 4;
      if ( !--v46 )
      {
        v24 = v38 - v5;
        goto LABEL_23;
      }
    }
    if ( v47 != v38 )
      goto LABEL_26;
LABEL_41:
    ++v39;
  }
  while ( v35 != v39 );
  return v37;
}
