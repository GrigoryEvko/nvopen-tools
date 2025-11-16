// Function: sub_1CB94D0
// Address: 0x1cb94d0
//
__int64 __fastcall sub_1CB94D0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  __int64 *v12; // rax
  __int64 v13; // rdi
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  size_t v16; // r13
  __int64 v17; // r15
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // r9d
  __int64 *v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  _BYTE *v27; // rsi
  _QWORD *v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // r9d
  __int64 *v33; // rcx
  unsigned __int8 *v34; // r8
  __int64 v35; // r15
  _BYTE *v36; // rdi
  __int64 *v37; // rdx
  _BYTE *v38; // rdi
  _BYTE *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v44; // rax
  _BYTE *v45; // rax
  __int64 *v46; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v47; // [rsp+8h] [rbp-78h]
  unsigned int v48; // [rsp+10h] [rbp-70h]
  __int64 *v49; // [rsp+10h] [rbp-70h]
  __int64 *v50; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v51; // [rsp+18h] [rbp-68h]
  unsigned int v52; // [rsp+18h] [rbp-68h]
  unsigned int v53; // [rsp+18h] [rbp-68h]
  _QWORD *v54; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v55; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v56; // [rsp+38h] [rbp-48h]
  _BYTE *v57; // [rsp+40h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 8);
  v55 = 0;
  v56 = 0;
  v57 = 0;
  if ( !v10 )
  {
    LODWORD(v10) = 0;
    return (unsigned int)v10;
  }
  do
  {
    while ( 1 )
    {
      v28 = sub_1648700(v10);
      v29 = v28[-3 * (*((_DWORD *)v28 + 5) & 0xFFFFFFF)];
      if ( *(_BYTE *)(v29 + 16) == 78 )
        v29 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      v30 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
        v12 = *(__int64 **)(v30 - 8);
      else
        v12 = (__int64 *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
      v13 = *v12;
      if ( *(_BYTE *)(*v12 + 16) == 3 )
        v13 = *(_QWORD *)(v13 - 24);
      v14 = (unsigned __int8 *)sub_1595920(v13);
      v16 = v15 - 1;
      v51 = v14;
      if ( !v15 )
        v16 = 0;
      v17 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      v18 = sub_16D1B30((__int64 *)a1, v14, v16);
      if ( v18 == -1 )
        v19 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      else
        v19 = *(_QWORD *)a1 + 8LL * v18;
      v20 = 0;
      if ( v19 != v17 )
      {
        v21 = sub_16D19C0(a1, v51, v16);
        v22 = (__int64 *)(*(_QWORD *)a1 + 8LL * v21);
        v23 = *v22;
        if ( *v22 )
        {
          if ( v23 != -8 )
          {
LABEL_13:
            v20 = *(int *)(v23 + 8);
            goto LABEL_14;
          }
          --*(_DWORD *)(a1 + 16);
        }
        v46 = v22;
        v48 = v21;
        v31 = malloc(v16 + 17);
        v32 = v48;
        v33 = v46;
        v34 = v51;
        v35 = v31;
        if ( !v31 )
        {
          if ( v16 == -17 )
          {
            v44 = malloc(1u);
            v32 = v48;
            v33 = v46;
            v34 = v51;
            if ( v44 )
            {
              v36 = (_BYTE *)(v44 + 16);
              v35 = v44;
              goto LABEL_43;
            }
          }
          v47 = v34;
          v50 = v33;
          v53 = v32;
          sub_16BD1C0("Allocation failed", 1u);
          v32 = v53;
          v33 = v50;
          v34 = v47;
        }
        v36 = (_BYTE *)(v35 + 16);
        if ( !v16 )
        {
LABEL_26:
          v36[v16] = 0;
          *(_QWORD *)v35 = v16;
          *(_DWORD *)(v35 + 8) = 0;
          *v33 = v35;
          ++*(_DWORD *)(a1 + 12);
          v37 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v32));
          v23 = *v37;
          if ( *v37 == -8 || !v23 )
          {
            do
            {
              do
              {
                v23 = v37[1];
                ++v37;
              }
              while ( !v23 );
            }
            while ( v23 == -8 );
          }
          goto LABEL_13;
        }
LABEL_43:
        v49 = v33;
        v52 = v32;
        v45 = memcpy(v36, v34, v16);
        v33 = v49;
        v32 = v52;
        v36 = v45;
        goto LABEL_26;
      }
LABEL_14:
      v24 = sub_15A0680(*v28, v20, 0);
      sub_164D160((__int64)v28, v24, a3, a4, a5, a6, v25, v26, a9, a10);
      v54 = v28;
      v27 = v56;
      if ( v56 != v57 )
        break;
      sub_17C2330((__int64)&v55, v56, &v54);
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_32;
    }
    if ( v56 )
    {
      *(_QWORD *)v56 = v28;
      v27 = v56;
    }
    v56 = v27 + 8;
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v10 );
LABEL_32:
  v38 = v55;
  v39 = (_BYTE *)(v57 - v55);
  v40 = (v56 - v55) >> 3;
  if ( v56 != v55 )
  {
    if ( (_DWORD)v40 )
    {
      v41 = 0;
      v42 = 8LL * (unsigned int)(v40 - 1);
      while ( 1 )
      {
        sub_15F20C0(*(_QWORD **)&v38[v41]);
        v38 = v55;
        if ( v41 == v42 )
          break;
        v41 += 8;
      }
    }
    LODWORD(v10) = 1;
    v39 = (_BYTE *)(v57 - v38);
  }
  if ( v38 )
    j_j___libc_free_0(v38, v39);
  return (unsigned int)v10;
}
