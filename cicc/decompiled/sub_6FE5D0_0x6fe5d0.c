// Function: sub_6FE5D0
// Address: 0x6fe5d0
//
__int64 __fastcall sub_6FE5D0(
        __int64 a1,
        __int64 *a2,
        _QWORD *a3,
        __int64 *a4,
        _QWORD *a5,
        __int64 *a6,
        __int64 **a7,
        _QWORD *a8,
        __m128i *a9)
{
  __int64 v12; // r15
  _QWORD *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 *v19; // r10
  __int64 v20; // rax
  __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r15
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // eax
  __int64 *v30; // r10
  __int64 v31; // rdi
  __int64 v32; // rax
  int v33; // eax
  __int64 *v34; // [rsp+8h] [rbp-48h]
  __int64 *v35; // [rsp+8h] [rbp-48h]
  __int64 *v36; // [rsp+8h] [rbp-48h]
  __int64 *v37; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]

  v12 = *(_QWORD *)a1;
  if ( a6 )
    *a6 = 0;
  if ( a7 )
  {
    *a7 = 0;
    if ( a2 )
    {
      v13 = (_QWORD *)a2[15];
      v14 = sub_6E3F00(v13[47], a1, (int)v13);
      v18 = (__int64)a4;
      v19 = a2;
      *a4 = v14;
      goto LABEL_6;
    }
    v22 = sub_6E3DA0(v12, 0);
    v13 = (_QWORD *)v22;
    if ( *(_BYTE *)(v12 + 24) != 5 )
    {
      v34 = *(__int64 **)(v12 + 72);
      v23 = sub_6E3F00(*(_QWORD *)(v22 + 376), a1, v22);
      v19 = v34;
      *a4 = v23;
      *a7 = 0;
      goto LABEL_20;
    }
LABEL_29:
    v36 = *(__int64 **)(v12 + 56);
    v29 = sub_8D3A70(*(_QWORD *)v12);
    v30 = v36;
    if ( v29 || (v33 = sub_8D3D40(*(_QWORD *)v12), v30 = v36, v33) )
      v31 = *(_QWORD *)v12;
    else
      v31 = v13[47];
    v37 = v30;
    v32 = sub_6E3F00(v31, a1, (int)v13);
    v19 = v37;
    *a4 = v32;
    if ( !a7 )
    {
LABEL_7:
      if ( !v19 )
        goto LABEL_11;
      v20 = v19[15];
      if ( v20 )
        goto LABEL_9;
      goto LABEL_10;
    }
LABEL_6:
    *a7 = v19;
    goto LABEL_7;
  }
  if ( a2 )
  {
    v13 = (_QWORD *)a2[15];
    v24 = sub_6E3F00(v13[47], a1, (int)v13);
    v15 = (__int64)a4;
    v19 = a2;
    *a4 = v24;
    v20 = a2[15];
    if ( v20 )
    {
LABEL_9:
      if ( *(_BYTE *)(v20 + 16) == 5 )
      {
        *a6 = sub_6FE2E0(*(_QWORD *)(v20 + 144), a1, v15, v18, v16, v17);
        goto LABEL_15;
      }
    }
LABEL_10:
    v19 = (__int64 *)sub_6E3F50((__int64)v19);
LABEL_11:
    if ( *(_WORD *)(a1 + 8) == 183 )
    {
      if ( !v12 )
      {
LABEL_14:
        *(_QWORD *)(a1 + 16) = v19;
        goto LABEL_15;
      }
      goto LABEL_13;
    }
LABEL_21:
    sub_6F85E0(v19, a1, 2, a8, a9, v17);
    goto LABEL_15;
  }
  v27 = sub_6E3DA0(v12, 0);
  v13 = (_QWORD *)v27;
  if ( *(_BYTE *)(v12 + 24) == 5 )
    goto LABEL_29;
  v35 = *(__int64 **)(v12 + 72);
  v28 = sub_6E3F00(*(_QWORD *)(v27 + 376), a1, v27);
  v19 = v35;
  *a4 = v28;
LABEL_20:
  if ( *(_WORD *)(a1 + 8) != 183 )
    goto LABEL_21;
LABEL_13:
  if ( (*(_BYTE *)(v12 + 26) & 1) == 0 )
    goto LABEL_14;
  v39 = (__int64)v19;
  v25 = sub_6E2F40(1);
  *(_QWORD *)(v25 + 24) = sub_690A60(v39, a1, v26);
  *a6 = v25;
LABEL_15:
  result = *(_QWORD *)((char *)v13 + 356);
  *a3 = result;
  if ( a5 )
  {
    result = v13[46];
    *a5 = result;
  }
  return result;
}
