// Function: sub_27A80C0
// Address: 0x27a80c0
//
void __fastcall sub_27A80C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rbx
  __int64 v16; // r15
  char v17; // al
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 i; // rax
  unsigned __int64 v26; // r15
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rbx
  bool v30; // zf
  __int64 v31; // rdx
  __int64 v32; // rax
  __int128 v33; // [rsp-10h] [rbp-80h]
  __int128 v34; // [rsp-10h] [rbp-80h]
  int *v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  unsigned __int64 v39; // [rsp+20h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  unsigned __int64 v41; // [rsp+28h] [rbp-48h]
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  __int64 v43; // [rsp+38h] [rbp-38h]

  v6 = a2 - a1;
  v36 = a3;
  v39 = a2;
  if ( (__int64)(a2 - a1) <= 256 )
    return;
  if ( !a3 )
  {
    v41 = a2;
    goto LABEL_22;
  }
  v35 = (int *)(a1 + 16);
  while ( 2 )
  {
    --v36;
    v7 = v39 - 16;
    v42 = a4;
    v8 = a1 + 16 * ((__int64)(((v39 - a1) >> 63) + ((__int64)(v39 - a1) >> 4)) >> 1);
    v43 = a5;
    v9 = sub_27A2220(&v42, v35, v8);
    v10 = v39 - 16;
    if ( !v9 )
    {
      if ( !(unsigned __int8)sub_27A2220(&v42, v35, v10) )
      {
        v30 = (unsigned __int8)sub_27A2220(&v42, (int *)v8, v7) == 0;
        v11 = *(_DWORD *)a1;
        if ( v30 )
          goto LABEL_7;
LABEL_29:
        *(_DWORD *)a1 = *(_DWORD *)(v39 - 16);
        v31 = *(_QWORD *)(v39 - 8);
        *(_DWORD *)(v39 - 16) = v11;
        v32 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = v31;
        *(_QWORD *)(v39 - 8) = v32;
        goto LABEL_8;
      }
      v11 = *(_DWORD *)a1;
LABEL_20:
      v21 = *(_DWORD *)(a1 + 16);
      *(_DWORD *)(a1 + 16) = v11;
      v22 = *(_QWORD *)(a1 + 8);
      *(_DWORD *)a1 = v21;
      v23 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 24) = v22;
      *(_QWORD *)(a1 + 8) = v23;
      goto LABEL_8;
    }
    if ( !(unsigned __int8)sub_27A2220(&v42, (int *)v8, v10) )
    {
      v30 = (unsigned __int8)sub_27A2220(&v42, v35, v7) == 0;
      v11 = *(_DWORD *)a1;
      if ( !v30 )
        goto LABEL_29;
      goto LABEL_20;
    }
    v11 = *(_DWORD *)a1;
LABEL_7:
    *(_DWORD *)a1 = *(_DWORD *)v8;
    v12 = *(_QWORD *)(v8 + 8);
    *(_DWORD *)v8 = v11;
    v13 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(v8 + 8) = v13;
LABEL_8:
    v14 = a1 + 16;
    v15 = v39;
    v42 = a4;
    v43 = a5;
    while ( 1 )
    {
      v41 = v14;
      if ( (unsigned __int8)sub_27A2220(&v42, (int *)v14, a1) )
        goto LABEL_14;
      v16 = v15 - 16;
      do
      {
        v15 = v16;
        v17 = sub_27A2220(&v42, (int *)a1, v16);
        v16 -= 16;
      }
      while ( v17 );
      if ( v14 >= v15 )
        break;
      v18 = *(_DWORD *)v14;
      *(_DWORD *)v14 = *(_DWORD *)v15;
      v19 = *(_QWORD *)(v15 + 8);
      *(_DWORD *)v15 = v18;
      v20 = *(_QWORD *)(v14 + 8);
      *(_QWORD *)(v14 + 8) = v19;
      *(_QWORD *)(v15 + 8) = v20;
LABEL_14:
      v14 += 16LL;
    }
    v6 = v14 - a1;
    sub_27A80C0(v14, v39, v36, a4, a5);
    if ( (__int64)(v14 - a1) > 256 )
    {
      if ( v36 )
      {
        v39 = v14;
        continue;
      }
LABEL_22:
      v24 = v6 >> 4;
      for ( i = (v24 - 2) >> 1; ; i = v40 - 1 )
      {
        v40 = i;
        *((_QWORD *)&v33 + 1) = a5;
        *(_QWORD *)&v33 = a4;
        sub_27A7EB0(a1, i, v24, *(_QWORD *)(a1 + 16 * i), *(_QWORD *)(a1 + 16 * i + 8), a6, v33);
        if ( !v40 )
          break;
      }
      v26 = v41 - 16;
      do
      {
        v27 = *(_QWORD *)v26;
        v28 = *(_QWORD *)(v26 + 8);
        v29 = v26 - a1;
        v26 -= 16LL;
        *(_DWORD *)(v26 + 16) = *(_DWORD *)a1;
        *(_QWORD *)(v26 + 24) = *(_QWORD *)(a1 + 8);
        *((_QWORD *)&v34 + 1) = a5;
        *(_QWORD *)&v34 = a4;
        sub_27A7EB0(a1, 0, v29 >> 4, v27, v28, a6, v34);
      }
      while ( v29 > 16 );
    }
    break;
  }
}
