// Function: sub_148B410
// Address: 0x148b410
//
__int64 __fastcall sub_148B410(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  __int64 v10; // r13
  __int64 v11; // rax
  char v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // al
  char v16; // al
  __int64 v17; // r13
  __int64 v18; // r11
  __int64 v19; // r13
  __int64 v20; // r13
  __int64 v21; // r12
  unsigned int v22; // r15d
  char v23; // al
  unsigned int v24; // [rsp+4h] [rbp-5Ch]
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  char v27; // [rsp+18h] [rbp-48h]
  char v28; // [rsp+22h] [rbp-3Eh]
  char v29; // [rsp+23h] [rbp-3Dh]
  unsigned int v30; // [rsp+24h] [rbp-3Ch]
  __int64 v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+28h] [rbp-38h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = 0;
  if ( a2 )
  {
    if ( (unsigned __int8)sub_1481140(a1, a3, a4, a5) )
      return 1;
    v30 = sub_15FF730(a3);
    if ( v30 == a3 )
    {
      v28 = 0;
      v29 = 0;
    }
    else
    {
      v29 = sub_1481140(a1, v30, a4, a5);
      v15 = sub_1481140(a1, 0x21u, a4, a5);
      v28 = v15;
      if ( v29 && v15 )
        return 1;
    }
    v31 = **(_QWORD **)(a2 + 32);
    v10 = sub_13FC470(a2);
    if ( !v10 )
    {
LABEL_36:
      v17 = *(_QWORD *)(a1 + 48);
      if ( !*(_BYTE *)(v17 + 184) )
        sub_14CDF70(*(_QWORD *)(a1 + 48));
      v18 = *(_QWORD *)(v17 + 8);
      v19 = 32LL * *(unsigned int *)(v17 + 16);
      v33 = v18 + v19;
      if ( v18 + v19 == v18 )
        return 0;
      v20 = v18;
      v24 = a3;
      v26 = a5;
      while ( 1 )
      {
        v21 = *(_QWORD *)(v20 + 16);
        if ( !v21 )
          goto LABEL_50;
        v22 = sub_15CCE20(*(_QWORD *)(a1 + 56), *(_QWORD *)(v20 + 16), **(_QWORD **)(a2 + 32));
        if ( !(_BYTE)v22 )
          goto LABEL_50;
        v25 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
        if ( (unsigned __int8)sub_148B0D0(a1, v24, a4, v26, v25, 0) )
          return v22;
        if ( v30 == v24 )
          goto LABEL_50;
        if ( v29 )
          break;
        v23 = sub_148B0D0(a1, v30, a4, v26, v25, 0);
        v29 = v23;
        if ( !v28 )
        {
          v28 = sub_148B0D0(a1, 0x21u, a4, v26, v25, 0);
          if ( !v29 )
            goto LABEL_50;
          goto LABEL_48;
        }
        if ( v23 )
          return v22;
LABEL_50:
        v20 += 32;
        if ( v33 == v20 )
          return 0;
      }
      if ( v28 )
        return v22;
      v28 = sub_148B0D0(a1, 0x21u, a4, v26, v25, 0);
LABEL_48:
      if ( v28 )
        return v22;
      v29 = v22;
      goto LABEL_50;
    }
    while ( 1 )
    {
      if ( sub_148B340(a1, v10, a3, a4, a5) )
        return 1;
      if ( v30 != a3 )
      {
        if ( v29 )
        {
          if ( v28 )
            return 1;
          v28 = sub_148B340(a1, v10, 0x21u, a4, a5);
LABEL_13:
          if ( v28 )
            return 1;
          v29 = 1;
          goto LABEL_15;
        }
        v29 = sub_148B340(a1, v10, v30, a4, a5);
        if ( v28 )
        {
          if ( v29 )
            return 1;
        }
        else
        {
          v28 = sub_148B340(a1, v10, 0x21u, a4, a5);
          if ( v29 )
            goto LABEL_13;
        }
      }
LABEL_15:
      v11 = sub_157EBA0(v10);
      if ( *(_BYTE *)(v11 + 16) == 26 && (*(_DWORD *)(v11 + 20) & 0xFFFFFFF) != 1 )
      {
        v12 = *(_QWORD *)(v11 - 24) != v31;
        v32 = *(_QWORD *)(v11 - 72);
        v27 = v12;
        if ( (unsigned __int8)sub_148B0D0(a1, a3, a4, a5, v32, v12) )
          return 1;
        if ( v30 != a3 )
        {
          if ( v29 )
          {
            if ( v28 )
              return 1;
            v28 = sub_148B0D0(a1, 0x21u, a4, a5, v32, v27);
LABEL_22:
            if ( v28 )
              return 1;
            v29 = 1;
            goto LABEL_24;
          }
          v16 = sub_148B0D0(a1, v30, a4, a5, v32, v27);
          v29 = v16;
          if ( v28 )
          {
            if ( v16 )
              return 1;
          }
          else
          {
            v28 = sub_148B0D0(a1, 0x21u, a4, a5, v32, v27);
            if ( v29 )
              goto LABEL_22;
          }
        }
      }
LABEL_24:
      v13 = sub_1457840(a1, v10);
      v31 = v14;
      v10 = v13;
      if ( !v13 )
        goto LABEL_36;
    }
  }
  return v5;
}
