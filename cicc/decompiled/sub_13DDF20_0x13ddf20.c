// Function: sub_13DDF20
// Address: 0x13ddf20
//
unsigned __int8 *__fastcall sub_13DDF20(int a1, unsigned __int8 *a2, unsigned __int8 *a3, _QWORD *a4, int a5)
{
  unsigned int v5; // r15d
  int v9; // eax
  int v10; // edx
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // rsi
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // rdx
  unsigned __int8 *result; // rax
  unsigned __int8 *v16; // r10
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // r14
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rax
  unsigned __int8 *v21; // rax
  unsigned __int8 *v22; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v23; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v24; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v25; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v26; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v27; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v28; // [rsp-40h] [rbp-40h]

  if ( !a5 )
    return 0;
  v5 = a5 - 1;
  v9 = a2[16];
  v10 = a3[16];
  if ( (unsigned __int8)v9 <= 0x17u || (unsigned int)(v9 - 35) > 0x11 )
  {
    if ( (unsigned __int8)v10 <= 0x17u || (unsigned int)(v10 - 35) > 0x11 )
      return 0;
    v16 = 0;
    goto LABEL_29;
  }
  if ( (unsigned __int8)v10 > 0x17u )
  {
    if ( (unsigned int)(v10 - 35) > 0x11 )
    {
      if ( a1 != v9 - 24 )
        goto LABEL_23;
      v12 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
      v27 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
      v14 = (unsigned __int8 *)sub_13DDBD0(a1, v12, a3, a4, v5);
      if ( !v14 )
        goto LABEL_23;
      v13 = 0;
      goto LABEL_7;
    }
    v16 = a2;
    if ( a1 == v9 - 24 )
    {
      v12 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
      v13 = a3;
      v27 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
      v14 = (unsigned __int8 *)sub_13DDBD0(a1, v12, a3, a4, v5);
      if ( !v14 )
        goto LABEL_10;
      goto LABEL_7;
    }
LABEL_29:
    v13 = a3;
    if ( a1 != a3[16] - 24 )
      goto LABEL_11;
    goto LABEL_30;
  }
  if ( a1 != v9 - 24 )
    return 0;
  v11 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
  v12 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
  v13 = 0;
  v27 = v11;
  v14 = (unsigned __int8 *)sub_13DDBD0(a1, v12, a3, a4, v5);
  if ( !v14 )
  {
LABEL_23:
    if ( ((1LL << a1) & 0x1C019800) != 0 )
    {
      v16 = a2;
      v13 = 0;
      goto LABEL_13;
    }
    return 0;
  }
LABEL_7:
  if ( v14 == v12 )
    return a2;
  result = (unsigned __int8 *)sub_13DDBD0(a1, v27, v14, a4, v5);
  if ( result )
    return result;
  if ( !v13 )
    goto LABEL_23;
LABEL_10:
  v16 = a2;
  if ( a1 != v13[16] - 24 )
    goto LABEL_11;
LABEL_30:
  v25 = v16;
  v22 = (unsigned __int8 *)*((_QWORD *)v13 - 3);
  v23 = (unsigned __int8 *)*((_QWORD *)v13 - 6);
  v20 = (unsigned __int8 *)sub_13DDBD0(a1, a2, v23, a4, v5);
  v16 = v25;
  if ( v20 )
  {
    if ( v20 == v23 )
      return a3;
    result = (unsigned __int8 *)sub_13DDBD0(a1, v20, v22, a4, v5);
    v16 = v25;
    if ( result )
      return result;
  }
LABEL_11:
  if ( ((1LL << a1) & 0x1C019800) == 0 )
    return 0;
  if ( !v16 )
    goto LABEL_14;
LABEL_13:
  if ( a1 != v16[16] - 24
    || (v24 = (unsigned __int8 *)*((_QWORD *)v16 - 3),
        v26 = (unsigned __int8 *)*((_QWORD *)v16 - 6),
        (v21 = (unsigned __int8 *)sub_13DDBD0(a1, a3, v26, a4, v5)) == 0) )
  {
LABEL_14:
    if ( v13 )
    {
      if ( a1 == v13[16] - 24 )
      {
        v17 = (unsigned __int8 *)*((_QWORD *)v13 - 6);
        v18 = (unsigned __int8 *)*((_QWORD *)v13 - 3);
        v28 = v17;
        v19 = (unsigned __int8 *)sub_13DDBD0(a1, v18, a2, a4, v5);
        if ( v19 )
        {
          if ( v19 != v18 )
            return (unsigned __int8 *)sub_13DDBD0(a1, v28, v19, a4, v5);
          return a3;
        }
      }
    }
    return 0;
  }
  if ( v21 == v26 )
    return a2;
  result = (unsigned __int8 *)sub_13DDBD0(a1, v21, v24, a4, v5);
  if ( !result )
    goto LABEL_14;
  return result;
}
