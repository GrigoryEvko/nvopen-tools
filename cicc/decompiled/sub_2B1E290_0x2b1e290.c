// Function: sub_2B1E290
// Address: 0x2b1e290
//
unsigned __int8 **__fastcall sub_2B1E290(unsigned __int8 **a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int8 **v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // rax
  _QWORD *v11; // rsi
  __int64 v12; // r9
  _BYTE **v14; // r15
  unsigned __int8 v15; // al
  __int64 v16; // rax
  _QWORD *v17; // rsi
  __int64 v18; // r9
  unsigned __int8 v19; // al
  __int64 v20; // rax
  _QWORD *v21; // rsi
  __int64 v22; // r9
  unsigned __int8 v23; // al
  __int64 v24; // rax
  _QWORD *v25; // rsi
  __int64 v26; // r9
  unsigned __int8 v27; // al
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30; // r9
  unsigned __int8 v31; // al
  __int64 v32; // rax
  _QWORD *v33; // rsi
  __int64 v34; // r9
  unsigned __int8 v35; // al
  __int64 v36; // rax
  _QWORD *v37; // rsi
  __int64 v38; // r9
  unsigned __int8 **v40; // [rsp+8h] [rbp-48h]
  __int64 v41[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a2 - (_QWORD)a1;
  v7 = v6 >> 3;
  if ( v6 >> 5 <= 0 )
  {
    v8 = a1;
LABEL_22:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (unsigned __int8 **)a2;
        goto LABEL_36;
      }
      v27 = **v8;
      if ( v27 != 13 )
      {
        if ( v27 <= 0x1Cu )
          return v8;
        v28 = *a3;
        v41[0] = (__int64)*v8;
        v29 = (_QWORD *)(*(_QWORD *)v28 + 8LL * *(unsigned int *)(v28 + 8));
        if ( v29 == sub_2B0BAC0(*(_QWORD **)v28, (__int64)v29, v41)
          && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v30) )
        {
          return v8;
        }
      }
      ++v8;
    }
    v31 = **v8;
    if ( v31 != 13 )
    {
      if ( v31 <= 0x1Cu )
        return v8;
      v32 = *a3;
      v41[0] = (__int64)*v8;
      v33 = (_QWORD *)(*(_QWORD *)v32 + 8LL * *(unsigned int *)(v32 + 8));
      if ( v33 == sub_2B0BAC0(*(_QWORD **)v32, (__int64)v33, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v34) )
      {
        return v8;
      }
    }
    ++v8;
LABEL_36:
    v35 = **v8;
    if ( v35 != 13 )
    {
      if ( v35 <= 0x1Cu )
        return v8;
      v36 = *a3;
      v41[0] = (__int64)*v8;
      v37 = (_QWORD *)(*(_QWORD *)v36 + 8LL * *(unsigned int *)(v36 + 8));
      if ( v37 == sub_2B0BAC0(*(_QWORD **)v36, (__int64)v37, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v38) )
      {
        return v8;
      }
    }
    return (unsigned __int8 **)a2;
  }
  v8 = a1;
  v40 = &a1[4 * (v6 >> 5)];
  while ( 1 )
  {
    v9 = **v8;
    if ( v9 != 13 )
    {
      if ( v9 <= 0x1Cu )
        return v8;
      v10 = *a3;
      v41[0] = (__int64)*v8;
      v11 = (_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8));
      if ( v11 == sub_2B0BAC0(*(_QWORD **)v10, (__int64)v11, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v12) )
      {
        return v8;
      }
    }
    v14 = v8 + 1;
    v15 = *v8[1];
    if ( v15 != 13 )
    {
      if ( v15 <= 0x1Cu )
        return v14;
      v16 = *a3;
      v41[0] = (__int64)v8[1];
      v17 = (_QWORD *)(*(_QWORD *)v16 + 8LL * *(unsigned int *)(v16 + 8));
      if ( v17 == sub_2B0BAC0(*(_QWORD **)v16, (__int64)v17, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v18) )
      {
        return v14;
      }
    }
    v14 = v8 + 2;
    v19 = *v8[2];
    if ( v19 != 13 )
    {
      if ( v19 <= 0x1Cu )
        return v14;
      v20 = *a3;
      v41[0] = (__int64)v8[2];
      v21 = (_QWORD *)(*(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8));
      if ( v21 == sub_2B0BAC0(*(_QWORD **)v20, (__int64)v21, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v22) )
      {
        return v14;
      }
    }
    v14 = v8 + 3;
    v23 = *v8[3];
    if ( v23 != 13 )
    {
      if ( v23 <= 0x1Cu )
        return v14;
      v24 = *a3;
      v41[0] = (__int64)v8[3];
      v25 = (_QWORD *)(*(_QWORD *)v24 + 8LL * *(unsigned int *)(v24 + 8));
      if ( v25 == sub_2B0BAC0(*(_QWORD **)v24, (__int64)v25, v41)
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))a4)(*(_QWORD *)(a4 + 8), v26) )
      {
        return v14;
      }
    }
    v8 += 4;
    if ( v40 == v8 )
    {
      v7 = (a2 - (__int64)v8) >> 3;
      goto LABEL_22;
    }
  }
}
