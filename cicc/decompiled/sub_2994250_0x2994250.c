// Function: sub_2994250
// Address: 0x2994250
//
_QWORD *__fastcall sub_2994250(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // r14
  int v9; // edx
  unsigned __int64 v10; // rax
  int v11; // esi
  __int64 v12; // r13
  unsigned int v13; // edx
  __int64 v14; // rcx
  unsigned int v15; // esi
  unsigned __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r10
  int v19; // r14d
  __int64 *v20; // rdi
  unsigned int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r9
  __int64 v24; // rbx
  _QWORD *v25; // rax
  _QWORD *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // rbx
  _QWORD *v30; // rax
  __int64 v31; // r14
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  char v35; // r8
  _QWORD *result; // rax
  int v37; // r9d
  int v38; // edx
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  unsigned __int64 v44; // [rsp+28h] [rbp-48h] BYREF
  __int64 *v45; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int16 v46; // [rsp+38h] [rbp-38h]

  v5 = *(unsigned int *)(a1 + 72);
  v6 = *(_QWORD *)(a1 + 64);
  v7 = *(_QWORD *)(a1 + 744);
  v8 = *(_QWORD **)(v6 + 8 * v5 - 8);
  v9 = *(_DWORD *)(a1 + 760);
  v10 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v9 )
    return (_QWORD *)sub_29946C0(a1, a2, a3);
  v11 = v9 - 1;
  v12 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v14 = *(_QWORD *)(v7 + 16LL * v13);
  if ( v10 != v14 )
  {
    v37 = 1;
    while ( v14 != -4096 )
    {
      v13 = v11 & (v37 + v13);
      v14 = *(_QWORD *)(v7 + 16LL * v13);
      if ( v10 == v14 )
        goto LABEL_3;
      ++v37;
    }
    return (_QWORD *)sub_29946C0(a1, a2, a3);
  }
LABEL_3:
  if ( !(unsigned __int8)sub_298A9F0(a1, v8) )
    v12 = sub_29941A0(a1, 1);
  v15 = *(_DWORD *)(a1 + 760);
  v16 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
  v44 = v16;
  v17 = v16;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 736);
    v45 = 0;
    goto LABEL_31;
  }
  v18 = *(_QWORD *)(a1 + 744);
  v19 = 1;
  v20 = 0;
  v21 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v22 = (__int64 *)(v18 + 16LL * v21);
  v23 = *v22;
  if ( v16 != *v22 )
  {
    while ( v23 != -4096 )
    {
      if ( !v20 && v23 == -8192 )
        v20 = v22;
      v21 = (v15 - 1) & (v19 + v21);
      v22 = (__int64 *)(v18 + 16LL * v21);
      v23 = *v22;
      if ( v16 == *v22 )
        goto LABEL_7;
      ++v19;
    }
    v39 = *(_DWORD *)(a1 + 752);
    if ( !v20 )
      v20 = v22;
    ++*(_QWORD *)(a1 + 736);
    v38 = v39 + 1;
    v45 = v20;
    if ( 4 * (v39 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 756) - v38 > v15 >> 3 )
        goto LABEL_33;
      sub_22E02D0(a1 + 736, v15);
LABEL_32:
      sub_27EFA30(a1 + 736, (__int64 *)&v44, &v45);
      v17 = v44;
      v20 = v45;
      v38 = *(_DWORD *)(a1 + 752) + 1;
LABEL_33:
      *(_DWORD *)(a1 + 752) = v38;
      if ( *v20 != -4096 )
        --*(_DWORD *)(a1 + 756);
      *v20 = v17;
      v24 = 0;
      v20[1] = 0;
      goto LABEL_8;
    }
LABEL_31:
    sub_22E02D0(a1 + 736, 2 * v15);
    goto LABEL_32;
  }
LABEL_7:
  v24 = v22[1];
LABEL_8:
  sub_29946C0(a1, 0, v24);
  while ( !*(_BYTE *)(a1 + 172) )
  {
    if ( sub_C8CA60(a1 + 144, v24) )
      goto LABEL_14;
LABEL_25:
    sub_2994250(a1, 0, v24);
  }
  v25 = *(_QWORD **)(a1 + 152);
  v26 = &v25[*(unsigned int *)(a1 + 164)];
  if ( v25 == v26 )
    goto LABEL_25;
  while ( *v25 != v24 )
  {
    if ( v26 == ++v25 )
      goto LABEL_25;
  }
LABEL_14:
  v27 = sub_29941A0(a1, 0);
  v28 = v27;
  if ( !*(_DWORD *)(a1 + 72) && a2 )
  {
    v29 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    sub_B1AEF0(*(_QWORD *)(a1 + 56), v29, v27);
    sub_2993400(a1, v28, v29);
  }
  else
  {
    v29 = sub_2993B60(a1, v27);
  }
  sub_B43C20((__int64)&v45, v28);
  v40 = *(_QWORD *)(a1 + 24);
  v42 = (__int64)v45;
  v43 = v46;
  v30 = sub_BD2C40(72, 3u);
  v31 = (__int64)v30;
  if ( v30 )
    sub_B4C9A0((__int64)v30, v29, v12, v40, 3u, v43, v42, v43);
  sub_2988C20(a1, v31, v28);
  v34 = *(unsigned int *)(a1 + 808);
  if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 812) )
  {
    sub_C8D5F0(a1 + 800, (const void *)(a1 + 816), v34 + 1, 8u, v32, v33);
    v34 = *(unsigned int *)(a1 + 808);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8 * v34) = v31;
  ++*(_DWORD *)(a1 + 808);
  sub_2993400(a1, v28, v12);
  v35 = sub_22DB400(*(_QWORD **)(a1 + 40), v29);
  result = 0;
  if ( v35 )
    result = sub_22DDF00(*(_QWORD **)(a1 + 40), v29);
  *(_QWORD *)(a1 + 912) = result;
  return result;
}
