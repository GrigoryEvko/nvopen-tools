// Function: sub_121DDC0
// Address: 0x121ddc0
//
__int64 __fastcall sub_121DDC0(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4)
{
  int v8; // eax
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  int *v12; // rax
  int v13; // edi
  __int64 v14; // r8
  __int64 v15; // rdi
  __int64 v16; // r9
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  _BOOL8 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rcx
  int v40; // r8d
  __int64 v41; // [rsp+0h] [rbp-70h]
  __int64 v42; // [rsp+0h] [rbp-70h]
  __int64 v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+8h] [rbp-68h]
  __int64 v45; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+8h] [rbp-68h]
  _QWORD v48[2]; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+20h] [rbp-50h]
  __int16 v50; // [rsp+30h] [rbp-40h]

  v8 = *(_DWORD *)(a1 + 136);
  v9 = *(_QWORD *)(a1 + 120);
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = (v8 - 1) & (37 * a2);
    v12 = (int *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
    {
LABEL_3:
      v14 = *((_QWORD *)v12 + 1);
      if ( v14 )
      {
LABEL_4:
        v15 = *(_QWORD *)a1;
        v50 = 2307;
        v48[0] = "%";
        v49 = a2;
        return sub_120A960(v15, a4, (__int64)v48, a3, v14);
      }
    }
    else
    {
      v18 = 1;
      while ( v13 != -1 )
      {
        v40 = v18 + 1;
        v11 = v10 & (v18 + v11);
        v12 = (int *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == a2 )
          goto LABEL_3;
        v18 = v40;
      }
    }
  }
  v19 = *(_QWORD *)(a1 + 80);
  v20 = a1 + 72;
  if ( !v19 )
    goto LABEL_15;
  v21 = a1 + 72;
  do
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(v19 + 16);
      v23 = *(_QWORD *)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) >= a2 )
        break;
      v19 = *(_QWORD *)(v19 + 24);
      if ( !v23 )
        goto LABEL_13;
    }
    v21 = v19;
    v19 = *(_QWORD *)(v19 + 16);
  }
  while ( v22 );
LABEL_13:
  if ( v21 == v20 || *(_DWORD *)(v21 + 32) > a2 )
  {
LABEL_15:
    v24 = *(unsigned __int8 *)(a3 + 8);
    if ( (_BYTE)v24 != 7 )
      goto LABEL_16;
LABEL_35:
    v35 = *(_QWORD *)a1 + 176LL;
    v48[0] = "invalid use of a non-first-class type";
    v50 = 259;
    sub_11FD800(v35, a4, (__int64)v48, 1);
    return 0;
  }
  v14 = *(_QWORD *)(v21 + 40);
  if ( v14 )
    goto LABEL_4;
  v24 = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)v24 == 7 )
    goto LABEL_35;
LABEL_16:
  if ( v24 == 13 )
    goto LABEL_35;
  if ( (_BYTE)v24 == 8 )
  {
    v36 = *(_QWORD *)(a1 + 8);
    v50 = 257;
    v45 = v36;
    v37 = sub_B2BE50(v36);
    v38 = sub_22077B0(80);
    v16 = v38;
    if ( v38 )
    {
      v39 = v45;
      v46 = v38;
      sub_AA4D50(v38, v37, (__int64)v48, v39, 0);
      v16 = v46;
    }
  }
  else
  {
    v50 = 257;
    v25 = sub_22077B0(40);
    v16 = v25;
    if ( v25 )
    {
      v43 = v25;
      sub_B2BA90(v25, a3, (__int64)v48, 0, 0);
      v16 = v43;
    }
  }
  v26 = *(_QWORD *)(a1 + 80);
  v27 = a1 + 72;
  if ( !v26 )
    goto LABEL_27;
  do
  {
    while ( 1 )
    {
      v28 = *(_QWORD *)(v26 + 16);
      v29 = *(_QWORD *)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) >= a2 )
        break;
      v26 = *(_QWORD *)(v26 + 24);
      if ( !v29 )
        goto LABEL_25;
    }
    v27 = v26;
    v26 = *(_QWORD *)(v26 + 16);
  }
  while ( v28 );
LABEL_25:
  if ( v20 == v27 || *(_DWORD *)(v27 + 32) > a2 )
  {
LABEL_27:
    v41 = v16;
    v44 = v27;
    v30 = sub_22077B0(56);
    *(_DWORD *)(v30 + 32) = a2;
    v27 = v30;
    *(_QWORD *)(v30 + 40) = 0;
    *(_QWORD *)(v30 + 48) = 0;
    v31 = sub_121DCC0((_QWORD *)(a1 + 64), v44, (unsigned int *)(v30 + 32));
    v33 = v41;
    if ( v32 )
    {
      v34 = v20 == v32 || v31 || a2 < *(_DWORD *)(v32 + 32);
      sub_220F040(v34, v27, v32, a1 + 72);
      ++*(_QWORD *)(a1 + 104);
      v16 = v41;
    }
    else
    {
      v42 = v31;
      v47 = v33;
      j_j___libc_free_0(v27, 56);
      v16 = v47;
      v27 = v42;
    }
  }
  *(_QWORD *)(v27 + 40) = v16;
  *(_QWORD *)(v27 + 48) = a4;
  return v16;
}
