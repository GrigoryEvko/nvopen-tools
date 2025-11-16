// Function: sub_E90E90
// Address: 0xe90e90
//
__int64 __fastcall sub_E90E90(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v4; // r9
  __int64 v6; // r13
  __int64 v7; // r12
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  unsigned int v12; // edi
  int v13; // eax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned __int64 v22; // r15
  unsigned __int64 i; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // rdx
  int v26; // ecx
  __int64 v27; // rcx
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // r15
  __int64 v37; // rdx
  int v38; // ecx
  __int64 v39; // r13
  __int64 v40; // rax
  int v41; // esi
  __int64 v42; // rax
  __int64 v43; // rsi
  int v44; // [rsp-58h] [rbp-58h] BYREF
  __int64 v45; // [rsp-50h] [rbp-50h]
  __int64 v46; // [rsp-48h] [rbp-48h]

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 384 )
    return result;
  v4 = a2;
  v6 = a1 + 24;
  v7 = a3;
  if ( !a3 )
  {
    v22 = a2;
    goto LABEL_33;
  }
  while ( 2 )
  {
    --v7;
    v8 = *(_QWORD *)(a1 + 32);
    v9 = a1
       + 8
       * ((__int64)(0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - a1) >> 3)) / 2
        + ((0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - a1) >> 3) + ((0xAAAAAAAAAAAAAAABLL * ((__int64)(v4 - a1) >> 3)) >> 63))
         & 0xFFFFFFFFFFFFFFFELL));
    v10 = *(_QWORD *)(v9 + 8);
    if ( v8 < v10 || v8 == v10 && *(_DWORD *)(a1 + 24) < *(_DWORD *)v9 )
    {
      v11 = *(_QWORD *)(v4 - 16);
      if ( v10 >= v11 && (v10 != v11 || *(_DWORD *)v9 >= *(_DWORD *)(v4 - 24)) )
      {
        if ( v8 >= v11 )
        {
          v29 = *(_DWORD *)(a1 + 24);
          if ( v8 != v11 || *(_DWORD *)(v4 - 24) <= v29 )
          {
            v41 = *(_DWORD *)a1;
            v14 = *(_QWORD *)(a1 + 8);
            *(_DWORD *)a1 = v29;
            v42 = *(_QWORD *)(a1 + 16);
            *(_QWORD *)(a1 + 8) = v8;
            *(_DWORD *)(a1 + 24) = v41;
            v43 = *(_QWORD *)(a1 + 40);
            *(_QWORD *)(a1 + 32) = v14;
            *(_QWORD *)(a1 + 16) = v43;
            *(_QWORD *)(a1 + 40) = v42;
            goto LABEL_9;
          }
        }
        goto LABEL_28;
      }
LABEL_24:
      v25 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v10;
      v26 = *(_DWORD *)v9;
      *(_QWORD *)(v9 + 8) = v25;
      LODWORD(v25) = *(_DWORD *)a1;
      *(_DWORD *)a1 = v26;
      v27 = *(_QWORD *)(v9 + 16);
      *(_DWORD *)v9 = v25;
      v28 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 16) = v27;
      *(_QWORD *)(v9 + 16) = v28;
      v14 = *(_QWORD *)(a1 + 32);
      v8 = *(_QWORD *)(a1 + 8);
      goto LABEL_9;
    }
    v11 = *(_QWORD *)(v4 - 16);
    if ( v8 >= v11 )
    {
      if ( v8 == v11 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( v12 < *(_DWORD *)(v4 - 24) )
          goto LABEL_8;
      }
      if ( v10 < v11 || v10 == v11 && *(_DWORD *)v9 < *(_DWORD *)(v4 - 24) )
      {
LABEL_28:
        v30 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = v11;
        v31 = *(_DWORD *)(v4 - 24);
        *(_QWORD *)(v4 - 16) = v30;
        LODWORD(v30) = *(_DWORD *)a1;
        *(_DWORD *)a1 = v31;
        v32 = *(_QWORD *)(v4 - 8);
        *(_DWORD *)(v4 - 24) = v30;
        v33 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 16) = v32;
        *(_QWORD *)(v4 - 8) = v33;
        v14 = *(_QWORD *)(a1 + 32);
        v8 = *(_QWORD *)(a1 + 8);
        goto LABEL_9;
      }
      goto LABEL_24;
    }
    v12 = *(_DWORD *)(a1 + 24);
LABEL_8:
    v13 = *(_DWORD *)a1;
    v14 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)a1 = v12;
    v15 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 8) = v8;
    *(_DWORD *)(a1 + 24) = v13;
    v16 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 32) = v14;
    *(_QWORD *)(a1 + 16) = v15;
    *(_QWORD *)(a1 + 40) = v16;
LABEL_9:
    v17 = v6;
    v18 = v4;
    while ( 1 )
    {
      v22 = v17;
      if ( v8 <= v14 && (v8 != v14 || *(_DWORD *)v17 >= *(_DWORD *)a1) )
        break;
LABEL_14:
      v14 = *(_QWORD *)(v17 + 32);
      v17 += 24LL;
    }
    for ( i = v18 - 24; ; i -= 24LL )
    {
      v24 = *(_QWORD *)(i + 8);
      v18 = i;
      if ( v24 <= v8 && (v24 != v8 || *(_DWORD *)a1 >= *(_DWORD *)i) )
        break;
    }
    if ( v17 < i )
    {
      *(_QWORD *)(v17 + 8) = v24;
      *(_QWORD *)(i + 8) = v14;
      v19 = *(_DWORD *)v17;
      *(_DWORD *)v17 = *(_DWORD *)i;
      v20 = *(_QWORD *)(i + 16);
      *(_DWORD *)i = v19;
      v21 = *(_QWORD *)(v17 + 16);
      *(_QWORD *)(v17 + 16) = v20;
      *(_QWORD *)(i + 16) = v21;
      v8 = *(_QWORD *)(a1 + 8);
      goto LABEL_14;
    }
    sub_E90E90(v17, v4, v7);
    result = v17 - a1;
    if ( (__int64)(v17 - a1) > 384 )
    {
      if ( v7 )
      {
        v4 = v17;
        continue;
      }
LABEL_33:
      v34 = v22;
      v35 = v22;
      v36 = v22 - 24;
      sub_E90D80(a1, v35, v34);
      do
      {
        v37 = *(_QWORD *)(v36 + 8);
        v38 = *(_DWORD *)v36;
        v39 = v36 - a1;
        v40 = *(_QWORD *)(v36 + 16);
        v36 -= 24LL;
        *(_QWORD *)(v36 + 32) = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(v36 + 24) = *(_DWORD *)a1;
        *(_QWORD *)(v36 + 40) = *(_QWORD *)(a1 + 16);
        v45 = v37;
        v44 = v38;
        v46 = v40;
        result = sub_E8F100(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v39 >> 3), &v44);
      }
      while ( v39 > 24 );
    }
    return result;
  }
}
