// Function: sub_2686160
// Address: 0x2686160
//
__int64 __fastcall sub_2686160(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  unsigned __int8 v4; // al
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 **v8; // rax
  __int64 *v9; // r9
  __int64 **i; // r15
  __int64 v11; // rax
  __int64 *v12; // rbx
  int v13; // r15d
  __int64 *v14; // r12
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r9
  int v30; // r11d
  __int64 *v31; // r10
  __int64 v32; // rbx
  int v33; // edi
  int v34; // ecx
  __int64 v35; // r13
  _QWORD *v36; // [rsp+0h] [rbp-100h]
  __int64 v37; // [rsp+8h] [rbp-F8h]
  __int64 **v38; // [rsp+10h] [rbp-F0h]
  char v40; // [rsp+4Fh] [rbp-B1h] BYREF
  unsigned int v41; // [rsp+50h] [rbp-B0h] BYREF
  int v42; // [rsp+54h] [rbp-ACh] BYREF
  __int64 v43; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v44[2]; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD v45[6]; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v46; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-58h]
  _BYTE v48[80]; // [rsp+B0h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 72);
  v41 = 1;
  v3 = v2 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v2 & 3) == 3 )
    v3 = *(_QWORD *)(v3 + 24);
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 )
  {
    if ( v4 == 22 )
    {
      v3 = *(_QWORD *)(v3 + 24);
    }
    else if ( v4 <= 0x1Cu )
    {
      v3 = 0;
    }
    else
    {
      v3 = sub_B43CB0(v3);
    }
  }
  v5 = *(int *)(a1 + 100);
  v6 = *(_QWORD *)(a2 + 208);
  v45[2] = a1;
  v45[0] = a2;
  v42 = v5;
  v7 = *(int *)(v6 + 72 * v5 + 34640);
  v37 = v5;
  v45[1] = &v42;
  v44[1] = (__int64)&v41;
  v45[4] = &v41;
  v46 = (__int64 *)v48;
  v36 = (_QWORD *)(a1 + 32 * v5 + 104);
  v44[0] = (__int64)v36;
  v45[3] = v36;
  v47 = 0x800000000LL;
  v8 = (__int64 **)sub_267FA80(v6 + 160 * v7 + 3512, v3);
  v9 = *v8;
  i = v8;
  v11 = (__int64)&(*v8)[*((unsigned int *)v8 + 2)];
  v12 = v9;
  if ( v9 == (__int64 *)v11 )
    goto LABEL_16;
  v38 = i;
  v13 = 0;
  v14 = (__int64 *)v11;
  do
  {
    while ( !(unsigned __int8)sub_2685EC0(v44, *v12) )
    {
      ++v12;
      ++v13;
      if ( v14 == v12 )
        goto LABEL_14;
    }
    v17 = (unsigned int)v47;
    v18 = (unsigned int)v47 + 1LL;
    if ( v18 > HIDWORD(v47) )
    {
      sub_C8D5F0((__int64)&v46, v48, v18, 4u, v15, v16);
      v17 = (unsigned int)v47;
    }
    ++v12;
    *((_DWORD *)v46 + v17) = v13++;
    LODWORD(v47) = v47 + 1;
  }
  while ( v14 != v12 );
LABEL_14:
  v19 = v47;
  for ( i = v38; (_DWORD)v47; v19 = v47 )
  {
    v20 = *((_DWORD *)v46 + v19 - 1);
    LODWORD(v47) = v19 - 1;
    (*i)[v20] = (*i)[(unsigned int)(*((_DWORD *)i + 2))-- - 1];
LABEL_16:
    ;
  }
  if ( v46 != (__int64 *)v48 )
    _libc_free((unsigned __int64)v46);
  v40 = 0;
  LODWORD(v46) = 56;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_26A3E20,
    (__int64)v45,
    a1,
    (int *)&v46,
    1,
    &v40,
    1,
    0);
  v21 = *(_QWORD *)(v3 + 80);
  if ( !v21 )
    BUG();
  v22 = *(_QWORD *)(v21 + 32);
  if ( v22 )
    v22 -= 24;
  v43 = v22;
  if ( !v41 )
  {
    v24 = a1 + 32 * v37;
    v25 = *(_DWORD *)(v24 + 128);
    if ( v25 )
    {
      v26 = *(_QWORD *)(v24 + 112);
      v27 = (v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v22 == *v28 )
        return v41;
      v30 = 1;
      v31 = 0;
      while ( v29 != -4096 )
      {
        if ( v29 != -8192 || v31 )
          v28 = v31;
        v27 = (v25 - 1) & (v30 + v27);
        v29 = *(_QWORD *)(v26 + 16LL * v27);
        if ( v22 == v29 )
          return v41;
        ++v30;
        v31 = v28;
        v28 = (__int64 *)(v26 + 16LL * v27);
      }
      if ( !v31 )
        v31 = v28;
      v32 = a1 + 32 * v37;
      ++*v36;
      v33 = *(_DWORD *)(v32 + 120);
      v46 = v31;
      v34 = v33 + 1;
      if ( 4 * (v33 + 1) < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(v32 + 124) - v34 <= v25 >> 3 )
        {
          sub_2685CE0((__int64)v36, v25);
          sub_2677F80((__int64)v36, &v43, &v46);
          v22 = v43;
          v31 = v46;
          v34 = *(_DWORD *)(v32 + 120) + 1;
        }
        goto LABEL_33;
      }
    }
    else
    {
      v46 = 0;
      ++*v36;
    }
    sub_2685CE0((__int64)v36, 2 * v25);
    sub_2677F80((__int64)v36, &v43, &v46);
    v22 = v43;
    v31 = v46;
    v34 = *(_DWORD *)(a1 + 32 * v37 + 120) + 1;
LABEL_33:
    v35 = 32 * v37 + a1;
    *(_DWORD *)(v35 + 120) = v34;
    if ( *v31 != -4096 )
      --*(_DWORD *)(v35 + 124);
    *v31 = v22;
    v31[1] = 0;
  }
  return v41;
}
