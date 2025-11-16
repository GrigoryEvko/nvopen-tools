// Function: sub_25E75F0
// Address: 0x25e75f0
//
__int64 __fastcall sub_25E75F0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v4; // r9
  _QWORD *v6; // r13
  __int64 v7; // r12
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // rdx
  unsigned __int64 v19; // r14
  _QWORD *v20; // rax
  unsigned __int64 i; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37[11]; // [rsp-58h] [rbp-58h] BYREF

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 384 )
    return result;
  v4 = (_QWORD *)a2;
  v6 = (_QWORD *)(a1 + 24);
  v7 = a3;
  if ( !a3 )
  {
    v19 = a2;
    goto LABEL_22;
  }
  while ( 2 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    v9 = *(_QWORD *)(a1 + 16);
    --v7;
    v10 = *(v4 - 1);
    v11 = (__int64 *)(a1
                    + 8
                    * (((__int64)(0xAAAAAAAAAAAAAAABLL * (result >> 3)) >> 1)
                     + ((0xAAAAAAAAAAAAAAABLL * (result >> 3)) & 0xFFFFFFFFFFFFFFFELL)));
    v12 = v11[2];
    if ( v8 >= v12 )
    {
      if ( v8 < v10 )
        goto LABEL_6;
      if ( v12 < v10 )
      {
LABEL_16:
        *(_QWORD *)(a1 + 16) = v10;
        v26 = *(v4 - 2);
        *(v4 - 1) = v9;
        v27 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = v26;
        v28 = *(v4 - 3);
        *(v4 - 2) = v27;
        v29 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v28;
        *(v4 - 3) = v29;
        v9 = *(_QWORD *)(a1 + 40);
        v8 = *(_QWORD *)(a1 + 16);
        goto LABEL_7;
      }
LABEL_20:
      *(_QWORD *)(a1 + 16) = v12;
      v30 = v11[1];
      v11[2] = v9;
      v31 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v30;
      v32 = *v11;
      v11[1] = v31;
      v33 = *(_QWORD *)a1;
      *(_QWORD *)a1 = v32;
      *v11 = v33;
      v9 = *(_QWORD *)(a1 + 40);
      v8 = *(_QWORD *)(a1 + 16);
      goto LABEL_7;
    }
    if ( v12 < v10 )
      goto LABEL_20;
    if ( v8 < v10 )
      goto LABEL_16;
LABEL_6:
    v13 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 16) = v8;
    *(_QWORD *)(a1 + 40) = v9;
    *(_QWORD *)(a1 + 8) = v14;
    v15 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = v13;
    v16 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v15;
    *(_QWORD *)(a1 + 24) = v16;
LABEL_7:
    v17 = v6;
    v18 = v4;
    while ( 1 )
    {
      v19 = (unsigned __int64)v17;
      if ( v8 > v9 )
        goto LABEL_13;
      v20 = v18 - 3;
      for ( i = *(v18 - 1); v8 < i; v20 -= 3 )
        i = *(v20 - 1);
      if ( v17 >= v20 )
        break;
      v17[2] = i;
      v22 = v20[1];
      v20[2] = v9;
      v23 = v17[1];
      v17[1] = v22;
      v24 = *v20;
      v20[1] = v23;
      v25 = *v17;
      *v17 = v24;
      *v20 = v25;
      v8 = *(_QWORD *)(a1 + 16);
      v18 = v20;
LABEL_13:
      v9 = v17[5];
      v17 += 3;
    }
    sub_25E75F0(v17, v4, v7);
    result = (__int64)v17 - a1;
    if ( (__int64)v17 - a1 > 384 )
    {
      if ( v7 )
      {
        v4 = v17;
        continue;
      }
LABEL_22:
      sub_25E74C0((char *)a1, (char *)v19, v19);
      do
      {
        v34 = *(_QWORD *)(v19 - 8);
        v19 -= 24LL;
        v35 = *(_QWORD *)(v19 + 8);
        v36 = *(_QWORD *)v19;
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v19 + 8) = *(_QWORD *)(a1 + 8);
        *(_QWORD *)v19 = *(_QWORD *)a1;
        v37[1] = v35;
        v37[0] = v36;
        v37[2] = v34;
        result = sub_25DC3F0(a1, 0, 0xAAAAAAAAAAAAAAABLL * ((__int64)(v19 - a1) >> 3), v37);
      }
      while ( (__int64)(v19 - a1) > 24 );
    }
    return result;
  }
}
