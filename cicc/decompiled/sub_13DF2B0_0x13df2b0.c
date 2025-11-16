// Function: sub_13DF2B0
// Address: 0x13df2b0
//
_QWORD *__fastcall sub_13DF2B0(int a1, unsigned __int8 *a2, unsigned __int8 *a3, int a4, _QWORD *a5, int a6)
{
  int v6; // r11d
  unsigned int v8; // r8d
  int v11; // eax
  int v12; // eax
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // rdx
  _QWORD *result; // rax
  unsigned __int8 *v16; // r14
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rdx
  unsigned __int8 *v20; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v21; // [rsp-58h] [rbp-58h]
  unsigned int v22; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v23; // [rsp-50h] [rbp-50h]
  int v24; // [rsp-44h] [rbp-44h]
  unsigned int v25; // [rsp-44h] [rbp-44h]
  unsigned __int8 *v26; // [rsp-40h] [rbp-40h]

  if ( !a6 )
    return 0;
  v6 = a1;
  v8 = a6 - 1;
  v11 = a2[16];
  if ( (unsigned __int8)v11 > 0x17u && (unsigned int)(v11 - 35) <= 0x11 && a4 == v11 - 24 )
  {
    v16 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
    v25 = a6 - 1;
    v23 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
    v17 = (unsigned __int8 *)sub_13DDBD0(a1, v16, a3, a5, v8);
    v6 = a1;
    v8 = v25;
    v21 = v17;
    if ( v17 )
    {
      v18 = (unsigned __int8 *)sub_13DDBD0(a1, v23, a3, a5, v25);
      v6 = a1;
      v8 = v25;
      v19 = v18;
      if ( v18 )
      {
        if ( v21 == v16 )
        {
          result = a2;
          if ( v23 == v19 )
            return result;
        }
        if ( ((1LL << a4) & 0x1C019800) != 0 && v23 == v21 )
        {
          result = a2;
          if ( v19 == v16 )
            return result;
        }
        result = sub_13DDBD0(a4, v21, v19, a5, v25);
        if ( result )
          return result;
        v8 = v25;
        v6 = a1;
      }
    }
  }
  v12 = a3[16];
  if ( (unsigned __int8)v12 <= 0x17u )
    return 0;
  if ( (unsigned int)(v12 - 35) > 0x11 )
    return 0;
  if ( a4 != v12 - 24 )
    return 0;
  v13 = (unsigned __int8 *)*((_QWORD *)a3 - 6);
  v22 = v8;
  v26 = (unsigned __int8 *)*((_QWORD *)a3 - 3);
  v24 = v6;
  v20 = (unsigned __int8 *)sub_13DDBD0(v6, a2, v13, a5, v8);
  if ( !v20 )
    return 0;
  v14 = (unsigned __int8 *)sub_13DDBD0(v24, a2, v26, a5, v22);
  if ( !v14 )
    return 0;
  if ( v20 != v13 || (result = a3, v14 != v26) )
  {
    if ( ((1LL << a4) & 0x1C019800) == 0 )
      return sub_13DDBD0(a4, v20, v14, a5, v22);
    if ( v20 != v26 )
      return sub_13DDBD0(a4, v20, v14, a5, v22);
    result = a3;
    if ( v14 != v13 )
      return sub_13DDBD0(a4, v20, v14, a5, v22);
  }
  return result;
}
