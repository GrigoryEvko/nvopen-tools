// Function: sub_DB8E00
// Address: 0xdb8e00
//
__int64 __fastcall sub_DB8E00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, unsigned __int8 a6)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  char v13; // al
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rcx
  unsigned __int8 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v24; // r15d
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rcx
  __int64 *v29; // rax
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  unsigned __int64 v32; // [rsp+10h] [rbp-40h]
  int v33; // [rsp+18h] [rbp-38h]

  v10 = sub_D47930(a3);
  if ( !v10 || !(unsigned __int8)sub_B19720(*(_QWORD *)(a2 + 40), a4, v10) )
    goto LABEL_14;
  v11 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == a4 + 48 )
    goto LABEL_35;
  if ( !v11 )
    BUG();
  v12 = v11 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_35:
    BUG();
  v13 = *(_BYTE *)(v11 - 24);
  if ( v13 != 31 )
  {
    if ( v13 == 32 )
    {
      v33 = sub_B46E30(v11 - 24);
      if ( !v33 )
      {
        sub_DED850(a1, a2, a3, v12, 0, a5);
        return a1;
      }
      v31 = 0;
      v24 = 0;
      while ( 1 )
      {
        v25 = sub_B46EC0(v12, v24);
        v26 = v25;
        if ( *(_BYTE *)(a3 + 84) )
        {
          v27 = *(_QWORD **)(a3 + 64);
          v28 = &v27[*(unsigned int *)(a3 + 76)];
          if ( v27 == v28 )
            goto LABEL_29;
          while ( v26 != *v27 )
          {
            if ( v28 == ++v27 )
              goto LABEL_29;
          }
LABEL_24:
          if ( v33 == ++v24 )
            goto LABEL_25;
        }
        else
        {
          v30 = v25;
          if ( sub_C8CA60(a3 + 56, v25) )
            goto LABEL_24;
          v26 = v30;
LABEL_29:
          if ( v31 )
            break;
          v31 = v26;
          if ( v33 == ++v24 )
          {
LABEL_25:
            sub_DED850(a1, a2, a3, v12, v31, a5);
            return a1;
          }
        }
      }
    }
LABEL_14:
    v18 = sub_D970F0(a2);
    sub_D97F80(a1, v18, v19, v20, v21, v22);
    return a1;
  }
  v14 = *(_QWORD *)(v11 - 56);
  if ( *(_BYTE *)(a3 + 84) )
  {
    v15 = *(_QWORD **)(a3 + 64);
    v16 = &v15[*(unsigned int *)(a3 + 76)];
    if ( v15 == v16 )
    {
LABEL_32:
      v17 = 1;
    }
    else
    {
      while ( v14 != *v15 )
      {
        if ( v16 == ++v15 )
          goto LABEL_32;
      }
      v17 = 0;
    }
  }
  else
  {
    v32 = v11;
    v29 = sub_C8CA60(a3 + 56, v14);
    v11 = v32;
    v17 = v29 == 0;
  }
  sub_DB8CC0(a1, a2, a3, *(_QWORD *)(v11 - 120), v17, a5, a6);
  return a1;
}
