// Function: sub_1E47810
// Address: 0x1e47810
//
__int64 __fastcall sub_1E47810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r12
  char v8; // r13
  unsigned int v9; // ebx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned int v16; // r12d
  __int64 v17; // rbx
  int v18; // r14d
  unsigned int v19; // r13d
  unsigned int v20; // eax
  unsigned int v21; // ebx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v30; // [rsp+20h] [rbp-D0h]
  __int64 v31; // [rsp+30h] [rbp-C0h]
  __int64 v33; // [rsp+40h] [rbp-B0h]
  __int64 v34; // [rsp+48h] [rbp-A8h]
  __int64 v35; // [rsp+50h] [rbp-A0h]
  unsigned int v36; // [rsp+50h] [rbp-A0h]
  __int64 *v37; // [rsp+60h] [rbp-90h]
  __int64 i; // [rsp+68h] [rbp-88h]
  __int64 v39; // [rsp+70h] [rbp-80h]
  int v40; // [rsp+7Ch] [rbp-74h]
  char v41; // [rsp+7Ch] [rbp-74h]
  int v42; // [rsp+88h] [rbp-68h] BYREF
  int v43; // [rsp+8Ch] [rbp-64h] BYREF
  _QWORD v44[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-50h]
  __int64 v46; // [rsp+A8h] [rbp-48h]
  int v47; // [rsp+B0h] [rbp-40h]

  v30 = (a3 - 1) / 2;
  if ( a2 < v30 )
  {
    for ( i = a2; ; i = v39 )
    {
      v42 = 0;
      v43 = 0;
      v31 = 2 * (i + 1);
      v37 = (__int64 *)(a1 + 8 * (v31 - 1));
      v39 = v31 - 1;
      v35 = *v37;
      v5 = *(_QWORD *)(*a5 + 96)
         + 10LL * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 16 * (i + 1)) + 16LL) + 6LL);
      v34 = *(_QWORD *)(*a5 + 96);
      v33 = *(_QWORD *)(*a5 + 72);
      v6 = v33 + 16LL * *(unsigned __int16 *)(v5 + 2);
      v7 = v33 + 16LL * *(unsigned __int16 *)(v5 + 4);
      if ( v6 == v7 )
      {
        v9 = -1;
      }
      else
      {
        v40 = 0;
        v8 = 0;
        v9 = -1;
        do
        {
          v10 = *(_DWORD *)(v6 + 4);
          v11 = sub_39FAC40(v10);
          if ( v11 < v9 )
          {
            v40 = v10;
            v9 = v11;
            v8 = 1;
          }
          v6 += 16;
        }
        while ( v7 != v6 );
        if ( v8 )
          v42 = v40;
      }
      v12 = v34 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(v35 + 16) + 6LL);
      v13 = 16LL * *(unsigned __int16 *)(v12 + 2);
      v14 = 16LL * *(unsigned __int16 *)(v12 + 4);
      v15 = v33 + v14;
      if ( v33 + v13 == v33 + v14 )
        goto LABEL_26;
      v41 = 0;
      v16 = -1;
      v36 = v9;
      v17 = v33 + v13;
      v18 = 0;
      do
      {
        v19 = *(_DWORD *)(v17 + 4);
        v20 = sub_39FAC40(v19);
        if ( v16 > v20 )
        {
          v41 = 1;
          v18 = v19;
          v16 = v20;
        }
        v17 += 16;
      }
      while ( v15 != v17 );
      if ( v41 )
        v43 = v18;
      if ( v36 != 1 || v16 != 1 )
        break;
      v21 = 0;
      if ( (unsigned __int8)sub_1932870((__int64)(a5 + 1), &v42, v44) )
        v21 = *(_DWORD *)(v44[0] + 4LL);
      if ( !(unsigned __int8)sub_1932870((__int64)(a5 + 1), &v43, v44) || *(_DWORD *)(v44[0] + 4LL) <= v21 )
        goto LABEL_26;
LABEL_23:
      *(_QWORD *)(a1 + 8 * i) = *v37;
      if ( v39 >= v30 )
        goto LABEL_28;
    }
    if ( v36 > v16 )
      goto LABEL_23;
LABEL_26:
    v37 = (__int64 *)(a1 + 16 * (i + 1));
    v39 = 2 * (i + 1);
    goto LABEL_23;
  }
  v39 = a2;
LABEL_28:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v39 )
  {
    v26 = v39;
    v27 = *(_QWORD *)(a1 + 8 * (2 * v39 + 1));
    v39 = 2 * v39 + 1;
    *(_QWORD *)(a1 + 8 * v26) = v27;
  }
  v44[1] = 1;
  v22 = a5[2];
  v23 = *a5;
  a5[2] = 0;
  ++a5[1];
  v45 = v22;
  v24 = a5[3];
  v44[0] = v23;
  LODWORD(v23) = *((_DWORD *)a5 + 8);
  a5[3] = 0;
  *((_DWORD *)a5 + 8) = 0;
  v46 = v24;
  v47 = v23;
  sub_1E47570(a1, v39, a2, a4, (__int64)v44);
  return j___libc_free_0(v45);
}
