// Function: sub_EF1680
// Address: 0xef1680
//
__int64 __fastcall sub_EF1680(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r13
  unsigned __int8 *v7; // rdx
  unsigned __int8 *v8; // rax
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int8 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int8 *v17; // rdx
  char *v18; // rsi
  unsigned __int8 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // eax
  unsigned __int8 *v26; // rax
  unsigned __int8 *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  unsigned __int8 *v44; // rax
  __int64 v45; // [rsp+8h] [rbp-D8h]
  __int64 v46; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v47; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v48[24]; // [rsp+20h] [rbp-C0h] BYREF

  v6 = a2;
  v7 = *(unsigned __int8 **)(a1 + 8);
  v8 = *(unsigned __int8 **)a1;
  if ( *(unsigned __int8 **)a1 == v7 )
  {
LABEL_4:
    v47 = 0;
    LOBYTE(v46) = 0;
    result = sub_EF53B0(a1, a2, &v46);
    v47 = result;
    if ( result )
    {
      v12 = *(unsigned __int8 **)a1;
      v13 = (unsigned __int8)v46;
      if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 && *v12 == 73 )
      {
        if ( !(_BYTE)v46 )
          sub_E18380(a1 + 296, &v47, (__int64)v12, 0, v10, v11);
        result = sub_EEFA10(a1, a2 != 0, (__int64)v12, v13, v10, v11);
        v48[0] = result;
        if ( result )
        {
          if ( a2 )
            *((_BYTE *)a2 + 1) = 1;
          return sub_EE7CC0(a1 + 808, &v47, (unsigned __int64 *)v48, v14, v15, v16);
        }
      }
      else if ( (_BYTE)v46 )
      {
        return 0;
      }
    }
    return result;
  }
  if ( *v8 != 78 )
  {
    if ( *v8 != 90 )
      goto LABEL_4;
    *(_QWORD *)a1 = v8 + 1;
    result = (__int64)sub_EF05F0((unsigned __int8 **)a1, 1);
    v46 = result;
    if ( !result )
      return result;
    v17 = *(unsigned __int8 **)a1;
    v18 = *(char **)(a1 + 8);
    result = 0;
    if ( *(char **)a1 == v18 || *v17 != 69 )
      return result;
    *(_QWORD *)a1 = v17 + 1;
    if ( v18 != (char *)(v17 + 1) && v17[1] == 115 )
    {
      *(_QWORD *)a1 = v17 + 2;
      *(_QWORD *)a1 = sub_E182C0((char *)v17 + 2, v18);
      result = sub_EE68C0(a1 + 808, "string literal");
      v48[0] = result;
      if ( result )
        return sub_EE7AF0(a1 + 808, &v46, v48, v41, v42, v43);
      return result;
    }
    sub_EE38E0(v48, (_QWORD *)a1);
    v19 = *(unsigned __int8 **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v19 == 100 )
    {
      v20 = 1;
      *(_QWORD *)a1 = v19 + 1;
      sub_EE32C0((char **)a1, 1);
      v44 = *(unsigned __int8 **)a1;
      if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v44 != 95 )
      {
        v21 = 0;
        goto LABEL_26;
      }
      v20 = (__int64)v6;
      *(_QWORD *)a1 = v44 + 1;
      v47 = sub_EF1680(a1, v6);
      if ( !v47 )
      {
        v21 = 0;
        goto LABEL_26;
      }
    }
    else
    {
      v20 = (__int64)v6;
      v21 = sub_EF1680(a1, v6);
      v47 = v21;
      if ( !v21 )
      {
LABEL_26:
        v45 = v21;
        sub_EE36A0(v48, (const void *)v20);
        return v45;
      }
      *(_QWORD *)a1 = sub_E182C0(*(char **)a1, *(char **)(a1 + 8));
    }
    v20 = (__int64)&v46;
    v21 = sub_EE7AF0(a1 + 808, &v46, &v47, v22, v23, v24);
    goto LABEL_26;
  }
  *(_QWORD *)a1 = v8 + 1;
  if ( v7 != v8 + 1 && v8[1] == 72 )
  {
    v27 = v8 + 2;
    *(_QWORD *)a1 = v27;
    if ( !a2 )
      goto LABEL_34;
    *((_BYTE *)a2 + 24) = 1;
    v27 = *(unsigned __int8 **)a1;
LABEL_60:
    v7 = *(unsigned __int8 **)(a1 + 8);
    goto LABEL_34;
  }
  v25 = sub_EE3340(a1);
  if ( !a2 )
  {
    v27 = *(unsigned __int8 **)a1;
    v7 = *(unsigned __int8 **)a1;
    if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)a1 )
      goto LABEL_34;
    if ( *v27 == 79 || *v27 == 82 )
    {
      ++v27;
      v7 = *(unsigned __int8 **)(a1 + 8);
      *(_QWORD *)a1 = v27;
      goto LABEL_34;
    }
    goto LABEL_60;
  }
  *((_DWORD *)a2 + 1) = v25;
  v26 = *(unsigned __int8 **)a1;
  if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) )
  {
    if ( *v26 == 79 )
    {
      *(_QWORD *)a1 = v26 + 1;
      *((_BYTE *)a2 + 8) = 2;
      v27 = *(unsigned __int8 **)a1;
      v7 = *(unsigned __int8 **)(a1 + 8);
      goto LABEL_34;
    }
    if ( *v26 == 82 )
    {
      *(_QWORD *)a1 = v26 + 1;
      *((_BYTE *)a2 + 8) = 1;
      v27 = *(unsigned __int8 **)a1;
      v7 = *(unsigned __int8 **)(a1 + 8);
      goto LABEL_34;
    }
  }
  *((_BYTE *)a2 + 8) = 0;
  v27 = *(unsigned __int8 **)a1;
  v7 = *(unsigned __int8 **)(a1 + 8);
LABEL_34:
  v47 = 0;
  while ( 1 )
  {
    if ( v7 == v27 )
    {
      if ( !v6 )
        goto LABEL_45;
      goto LABEL_38;
    }
    v28 = *v27;
    if ( (_BYTE)v28 == 69 )
      break;
    if ( !v6 )
    {
      v30 = v7 - v27;
      goto LABEL_40;
    }
LABEL_38:
    *((_BYTE *)v6 + 1) = 0;
    v29 = *(_QWORD *)(a1 + 8);
    v27 = *(unsigned __int8 **)a1;
    if ( v29 == *(_QWORD *)a1 )
      goto LABEL_45;
    v28 = *v27;
    v30 = v29 - (_QWORD)v27;
LABEL_40:
    if ( (_BYTE)v28 == 84 )
    {
      if ( v47 )
        return 0;
      v32 = sub_EEA020(a1, (__int64)a2, v30, v28, a5);
      v47 = v32;
      goto LABEL_47;
    }
    if ( (_BYTE)v28 == 73 )
    {
      if ( !v47 )
        return 0;
      v48[0] = sub_EEFA10(a1, v6 != 0, v30, v28, a5, a6);
      if ( !v48[0] || *(_BYTE *)(v47 + 8) == 45 )
        return 0;
      if ( v6 )
        *((_BYTE *)v6 + 1) = 1;
      v32 = sub_EE7CC0(a1 + 808, &v47, (unsigned __int64 *)v48, v37, v38, v39);
      v47 = v32;
      goto LABEL_47;
    }
    if ( v30 > 1 && (_BYTE)v28 == 68 )
    {
      if ( (v27[1] & 0xDF) != 0x54 )
      {
LABEL_45:
        v31 = 0;
LABEL_46:
        v32 = sub_EF4820(a1, v6, v47, v31);
        v47 = v32;
        goto LABEL_47;
      }
      if ( v47 )
        return 0;
      v32 = sub_EED1D0((_QWORD *)a1);
      v47 = v32;
LABEL_47:
      if ( !v32 )
        return 0;
      a2 = &v47;
      sub_E18380(a1 + 296, &v47, v33, v34, v35, v36);
      v27 = *(unsigned __int8 **)a1;
      v7 = *(unsigned __int8 **)(a1 + 8);
      if ( *(unsigned __int8 **)a1 != v7 && *v27 == 77 )
        *(_QWORD *)a1 = ++v27;
    }
    else
    {
      if ( (_BYTE)v28 != 83 )
        goto LABEL_45;
      if ( v30 > 1 && v27[1] == 116 )
      {
        a2 = (__int64 *)"std";
        *(_QWORD *)a1 = v27 + 2;
        v31 = sub_EE68C0(a1 + 808, "std");
      }
      else
      {
        v31 = sub_EE9AE0(a1, (__int64)a2, v30, v28, a5, a6);
      }
      if ( !v31 )
        return 0;
      if ( *(_BYTE *)(v31 + 8) == 27 )
        goto LABEL_46;
      if ( v47 )
        return 0;
      v27 = *(unsigned __int8 **)a1;
      v7 = *(unsigned __int8 **)(a1 + 8);
      v47 = v31;
    }
  }
  *(_QWORD *)a1 = v27 + 1;
  result = v47;
  if ( v47 )
  {
    v40 = *(_QWORD *)(a1 + 304);
    if ( *(_QWORD *)(a1 + 296) != v40 )
    {
      *(_QWORD *)(a1 + 304) = v40 - 8;
      return result;
    }
  }
  return 0;
}
