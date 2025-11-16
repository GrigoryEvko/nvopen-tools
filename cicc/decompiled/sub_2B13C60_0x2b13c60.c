// Function: sub_2B13C60
// Address: 0x2b13c60
//
__int64 __fastcall sub_2B13C60(_QWORD **a1, __int64 a2, int *a3)
{
  _QWORD **v3; // r10
  int *v4; // r11
  __int64 v5; // r12
  __int64 v6; // rax
  _QWORD **v7; // r12
  _QWORD *v8; // rdi
  _QWORD *v9; // rbx
  __int64 v10; // r10
  int *v11; // r11
  _QWORD *v12; // rdi
  _QWORD *v13; // rbx
  __int64 v14; // r10
  int *v15; // r11
  _QWORD *v16; // rdi
  _QWORD *v17; // rbx
  __int64 v18; // r10
  _QWORD *v19; // rbx
  __int64 v20; // r10
  int *v21; // r11
  __int64 result; // rax
  __int64 v23; // rbx
  _QWORD *v24; // r8
  __int64 v25; // r10
  __int64 v26; // rbx
  _QWORD *v27; // r8
  __int64 v28; // r10
  _QWORD *v29; // rbx
  bool v30; // zf
  __int64 v31; // r10

  v3 = a1;
  v4 = a3;
  v5 = (a2 - (__int64)a1) >> 8;
  v6 = (a2 - (__int64)a1) >> 6;
  if ( v5 <= 0 )
  {
LABEL_11:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return a2;
LABEL_19:
        v29 = &(*v3)[*((unsigned int *)v3 + 2)];
        v30 = v29 == sub_2B0A300(*v3, (__int64)v29, v4);
        result = v31;
        if ( v30 )
          return a2;
        return result;
      }
      v23 = (__int64)&(*v3)[*((unsigned int *)v3 + 2)];
      v24 = sub_2B0A300(*v3, v23, v4);
      result = v25;
      if ( (_QWORD *)v23 != v24 )
        return result;
      v3 = (_QWORD **)(v25 + 64);
    }
    v26 = (__int64)&(*v3)[*((unsigned int *)v3 + 2)];
    v27 = sub_2B0A300(*v3, v26, v4);
    result = v28;
    if ( (_QWORD *)v26 != v27 )
      return result;
    v3 = (_QWORD **)(v28 + 64);
    goto LABEL_19;
  }
  v7 = &a1[32 * v5];
  while ( 1 )
  {
    v19 = &(*v3)[*((unsigned int *)v3 + 2)];
    if ( v19 != sub_2B0A300(*v3, (__int64)v19, v4) )
      return v20;
    v8 = *(_QWORD **)(v20 + 64);
    v9 = &v8[*(unsigned int *)(v20 + 72)];
    if ( v9 != sub_2B0A300(v8, (__int64)v9, v21) )
      return v10 + 64;
    v12 = *(_QWORD **)(v10 + 128);
    v13 = &v12[*(unsigned int *)(v10 + 136)];
    if ( v13 != sub_2B0A300(v12, (__int64)v13, v11) )
      return v14 + 128;
    v16 = *(_QWORD **)(v14 + 192);
    v17 = &v16[*(unsigned int *)(v14 + 200)];
    if ( v17 != sub_2B0A300(v16, (__int64)v17, v15) )
      return v18 + 192;
    v3 = (_QWORD **)(v18 + 256);
    if ( v7 == v3 )
    {
      v6 = (a2 - (__int64)v3) >> 6;
      goto LABEL_11;
    }
  }
}
