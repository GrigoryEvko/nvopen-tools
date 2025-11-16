// Function: sub_3897BF0
// Address: 0x3897bf0
//
__int64 __fastcall sub_3897BF0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r13
  unsigned int v12; // r14d
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // r8d
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // di
  __int64 v28; // r8
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // r10
  __int64 v36; // r8
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rsi
  unsigned __int64 v43; // rdi
  unsigned int v44; // [rsp+Ch] [rbp-64h]
  _QWORD *v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+20h] [rbp-50h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v51; // [rsp+28h] [rbp-48h]
  unsigned int v52; // [rsp+34h] [rbp-3Ch] BYREF
  unsigned int *v53[7]; // [rsp+38h] [rbp-38h] BYREF

  v11 = *(_QWORD *)(a1 + 56);
  v52 = 0;
  v12 = sub_388BA90(a1, &v52);
  if ( (_BYTE)v12 )
    return v12;
  v13 = *(_QWORD *)(a1 + 824);
  v14 = a1 + 816;
  if ( !v13 )
    goto LABEL_9;
  v15 = a1 + 816;
  v16 = *(_QWORD *)(a1 + 824);
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v16 + 16);
      v18 = *(_QWORD *)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) >= v52 )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      if ( !v18 )
        goto LABEL_7;
    }
    v15 = v16;
    v16 = *(_QWORD *)(v16 + 16);
  }
  while ( v17 );
LABEL_7:
  if ( v14 == v15 || v52 < *(_DWORD *)(v15 + 32) )
  {
LABEL_9:
    v19 = *(_QWORD *)(a1 + 872);
    if ( v19 )
    {
      v20 = a1 + 864;
      do
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(v19 + 16);
          v22 = *(_QWORD *)(v19 + 24);
          if ( *(_DWORD *)(v19 + 32) >= v52 )
            break;
          v19 = *(_QWORD *)(v19 + 24);
          if ( !v22 )
            goto LABEL_14;
        }
        v20 = v19;
        v19 = *(_QWORD *)(v19 + 16);
      }
      while ( v21 );
LABEL_14:
      if ( a1 + 864 != v20 && v52 >= *(_DWORD *)(v20 + 32) )
      {
LABEL_32:
        v48 = v20;
        v30 = sub_1627350(*(__int64 **)a1, 0, 0, 2, 1);
        v35 = v48;
        v36 = v30;
        v37 = *(_QWORD *)(v48 + 40);
        *(_QWORD *)(v48 + 40) = v30;
        if ( v37 )
        {
          sub_16307F0(v37, 0, v31, v32, v30, a3, a4, a5, a6, v33, v34, a9, a10);
          v35 = v48;
          v36 = *(_QWORD *)(v48 + 40);
        }
        *(_QWORD *)(v35 + 48) = v11;
        *a2 = v36;
        v38 = *(_QWORD *)(a1 + 824);
        if ( v38 )
        {
          v39 = a1 + 816;
          do
          {
            while ( 1 )
            {
              v40 = *(_QWORD *)(v38 + 16);
              v41 = *(_QWORD *)(v38 + 24);
              if ( *(_DWORD *)(v38 + 32) >= v52 )
                break;
              v38 = *(_QWORD *)(v38 + 24);
              if ( !v41 )
                goto LABEL_39;
            }
            v39 = v38;
            v38 = *(_QWORD *)(v38 + 16);
          }
          while ( v40 );
LABEL_39:
          if ( v14 != v39 && v52 >= *(_DWORD *)(v39 + 32) )
            goto LABEL_42;
        }
        else
        {
          v39 = a1 + 816;
        }
        v53[0] = &v52;
        v39 = sub_38979A0((_QWORD *)(a1 + 808), v39, v53);
        v36 = *a2;
LABEL_42:
        v42 = *(_QWORD *)(v39 + 40);
        if ( v42 )
        {
          v51 = v36;
          sub_161E7C0(v39 + 40, v42);
          v36 = v51;
        }
        *(_QWORD *)(v39 + 40) = v36;
        if ( v36 )
          sub_1623A60(v39 + 40, v36, 2);
        return v12;
      }
    }
    else
    {
      v20 = a1 + 864;
    }
    v46 = v20;
    v45 = (_QWORD *)(a1 + 864);
    v23 = sub_22077B0(0x38u);
    v24 = v52;
    *(_QWORD *)(v23 + 40) = 0;
    *(_DWORD *)(v23 + 32) = v24;
    *(_QWORD *)(v23 + 48) = 0;
    v44 = v24;
    v47 = v23;
    v25 = sub_3897AF0((_QWORD *)(a1 + 856), v46, (unsigned int *)(v23 + 32));
    if ( v26 )
    {
      v27 = v25 || v45 == (_QWORD *)v26 || v44 < *(_DWORD *)(v26 + 32);
      sub_220F040(v27, v47, (_QWORD *)v26, v45);
      ++*(_QWORD *)(a1 + 896);
      v20 = v47;
    }
    else
    {
      v43 = v47;
      v49 = v25;
      j_j___libc_free_0(v43);
      v20 = v49;
    }
    goto LABEL_32;
  }
  v28 = a1 + 816;
  do
  {
    if ( v52 > *(_DWORD *)(v13 + 32) )
    {
      v13 = *(_QWORD *)(v13 + 24);
    }
    else
    {
      v28 = v13;
      v13 = *(_QWORD *)(v13 + 16);
    }
  }
  while ( v13 );
  if ( v14 == v28 || v52 < *(_DWORD *)(v28 + 32) )
  {
    v53[0] = &v52;
    v28 = sub_38979A0((_QWORD *)(a1 + 808), v28, v53);
  }
  *a2 = *(_QWORD *)(v28 + 40);
  return v12;
}
