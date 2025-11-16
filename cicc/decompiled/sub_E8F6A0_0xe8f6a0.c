// Function: sub_E8F6A0
// Address: 0xe8f6a0
//
__int64 *__fastcall sub_E8F6A0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 *a5)
{
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 i; // r12
  unsigned int v8; // r9d
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 **v11; // r8
  __int64 **v12; // r12
  _QWORD *v13; // r10
  __int64 *v14; // r9
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // r15
  unsigned int v18; // edx
  __int64 v19; // rax
  __int64 **v20; // rbx
  __int64 *v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // rdx
  __int64 **v29; // r12
  __int64 *v31; // [rsp+8h] [rbp-58h]
  _QWORD *v32; // [rsp+8h] [rbp-58h]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  _QWORD *v35; // [rsp+18h] [rbp-48h]
  _QWORD *v36; // [rsp+20h] [rbp-40h]
  unsigned int v37; // [rsp+20h] [rbp-40h]
  __int64 *v38; // [rsp+20h] [rbp-40h]
  _QWORD *v39; // [rsp+20h] [rbp-40h]
  __int64 **v40; // [rsp+28h] [rbp-38h]
  _QWORD *v41; // [rsp+28h] [rbp-38h]
  unsigned int v42; // [rsp+28h] [rbp-38h]

  v5 = (a3 - 1) / 2;
  if ( a2 < v5 )
  {
    v6 = v5;
    for ( i = a2; ; i = v10 )
    {
      v10 = 2 * (i + 1);
      v11 = (__int64 **)(a1 + 32 * (i + 1));
      v13 = *(v11 - 2);
      v14 = *v11;
      v15 = **v11;
      if ( v15 )
        break;
      v34 = a4;
      v36 = *(v11 - 2);
      if ( (*((_BYTE *)v14 + 9) & 0x70) != 0x20 || *((char *)v14 + 8) < 0 )
        BUG();
      *((_BYTE *)v14 + 8) |= 8u;
      v31 = v14;
      v16 = sub_E807D0(v14[3]);
      v13 = v36;
      a4 = v34;
      v11 = (__int64 **)(a1 + 32 * (i + 1));
      *v31 = (__int64)v16;
      v8 = *(_DWORD *)(v16[1] + 36LL);
      v9 = (_QWORD *)*v36;
      if ( !*v36 )
        goto LABEL_12;
LABEL_4:
      if ( v8 < *(_DWORD *)(v9[1] + 36LL) )
      {
        --v10;
        v11 = (__int64 **)(a1 + 16 * v10);
      }
      v12 = (__int64 **)(a1 + 16 * i);
      *v12 = *v11;
      v12[1] = v11[1];
      if ( v10 >= v6 )
      {
        if ( (a3 & 1) != 0 )
          goto LABEL_16;
        goto LABEL_33;
      }
    }
    v8 = *(_DWORD *)(*(_QWORD *)(v15 + 8) + 36LL);
    v9 = (_QWORD *)*v13;
    if ( *v13 )
      goto LABEL_4;
LABEL_12:
    v35 = a4;
    v37 = v8;
    v40 = v11;
    if ( (*((_BYTE *)v13 + 9) & 0x70) != 0x20 || *((char *)v13 + 8) < 0 )
      BUG();
    *((_BYTE *)v13 + 8) |= 8u;
    v32 = v13;
    v9 = sub_E807D0(v13[3]);
    a4 = v35;
    v8 = v37;
    v11 = v40;
    *v32 = v9;
    goto LABEL_4;
  }
  if ( (a3 & 1) != 0 )
  {
    v29 = (__int64 **)(a1 + 16 * a2);
    goto LABEL_30;
  }
  v10 = a2;
LABEL_33:
  if ( (a3 - 2) / 2 == v10 )
  {
    v26 = v10 + 1;
    v27 = (_QWORD *)(a1 + 16 * v10);
    v28 = (_QWORD *)(a1 + 32 * v26 - 16);
    *v27 = *v28;
    v27[1] = v28[1];
    v10 = 2 * v26 - 1;
  }
LABEL_16:
  v17 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v29 = (__int64 **)(a1 + 16 * v17);
      v21 = *v29;
      v22 = **v29;
      if ( v22 )
      {
        v18 = *(_DWORD *)(*(_QWORD *)(v22 + 8) + 36LL);
        v19 = *a4;
        if ( *a4 )
          goto LABEL_19;
      }
      else
      {
        v41 = a4;
        if ( (*((_BYTE *)v21 + 9) & 0x70) != 0x20 || *((char *)v21 + 8) < 0 )
          BUG();
        *((_BYTE *)v21 + 8) |= 8u;
        v38 = v21;
        v23 = sub_E807D0(v21[3]);
        a4 = v41;
        *v38 = (__int64)v23;
        v18 = *(_DWORD *)(v23[1] + 36LL);
        v19 = *v41;
        if ( *v41 )
        {
LABEL_19:
          v20 = (__int64 **)(a1 + 16 * v10);
          if ( v18 >= *(_DWORD *)(*(_QWORD *)(v19 + 8) + 36LL) )
            goto LABEL_29;
          goto LABEL_20;
        }
      }
      v42 = v18;
      if ( (*((_BYTE *)a4 + 9) & 0x70) != 0x20 || *((char *)a4 + 8) < 0 )
        BUG();
      *((_BYTE *)a4 + 8) |= 8u;
      v39 = a4;
      v20 = (__int64 **)(a1 + 16 * v10);
      v24 = sub_E807D0(a4[3]);
      a4 = v39;
      *v39 = v24;
      if ( v42 >= *(_DWORD *)(v24[1] + 36LL) )
      {
LABEL_29:
        v29 = v20;
        goto LABEL_30;
      }
LABEL_20:
      *v20 = *v29;
      v20[1] = v29[1];
      v10 = v17;
      if ( a2 >= v17 )
        goto LABEL_30;
      v17 = (v17 - 1) / 2;
    }
  }
  v29 = (__int64 **)(a1 + 16 * v10);
LABEL_30:
  *v29 = a4;
  v29[1] = a5;
  return a5;
}
