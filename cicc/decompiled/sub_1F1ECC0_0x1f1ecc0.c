// Function: sub_1F1ECC0
// Address: 0x1f1ecc0
//
__int64 __fastcall sub_1F1ECC0(__int64 *a1, unsigned int a2)
{
  __int64 v3; // r15
  _QWORD *v4; // r12
  __int64 v6; // rax
  int v7; // r9d
  unsigned int v8; // edx
  unsigned int v9; // r14d
  __int64 v10; // r15
  unsigned int v11; // eax
  unsigned int v12; // r14d
  __int64 v13; // rax
  unsigned int v14; // edx
  int v15; // r9d
  char v16; // si
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  int v19; // esi
  bool v20; // zf
  unsigned int v21; // edx
  __int64 v22; // r14
  unsigned int v23; // eax
  _QWORD *v24; // r13
  unsigned int i; // r12d
  int v26; // r9d
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 *v31; // rax
  unsigned int v32; // eax
  _QWORD *v33; // r12
  unsigned int v34; // r13d
  unsigned int j; // r15d
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-A0h]
  char v44; // [rsp+Fh] [rbp-91h]
  int v45; // [rsp+10h] [rbp-90h]
  unsigned __int8 v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+20h] [rbp-80h]
  unsigned int v49; // [rsp+28h] [rbp-78h]
  unsigned int v50; // [rsp+28h] [rbp-78h]
  int v51; // [rsp+2Ch] [rbp-74h]
  unsigned int v52; // [rsp+2Ch] [rbp-74h]
  unsigned int v53[4]; // [rsp+30h] [rbp-70h] BYREF
  _DWORD v54[4]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v55[10]; // [rsp+50h] [rbp-50h] BYREF

  v3 = 16LL * a2;
  v4 = a1 + 1;
  v51 = *(_DWORD *)(a1[1] + v3 + 12);
  v6 = sub_3945DA0(a1 + 1, a2);
  v7 = v51;
  v48 = v6;
  if ( v6 )
  {
    v52 = 2;
    v8 = 1;
    v55[0] = v6 & 0xFFFFFFFFFFFFFFC0LL;
    v9 = (v6 & 0x3F) + 1;
    v53[0] = v9;
    v7 += v9;
  }
  else
  {
    v52 = 1;
    v9 = 0;
    v8 = 0;
  }
  v10 = a1[1] + v3;
  v11 = *(_DWORD *)(v10 + 8);
  v45 = v7;
  v49 = v8;
  v53[v8] = v11;
  v12 = v11 + v9;
  v47 = v8;
  v55[v8] = *(_QWORD *)v10;
  v13 = sub_3945FF0(v4, a2);
  v14 = v49;
  v15 = v45;
  if ( v13 )
  {
    v16 = v13;
    v17 = v13 & 0xFFFFFFFFFFFFFFC0LL;
    v18 = v52;
    v50 = v49 + 2;
    v19 = (v16 & 0x3F) + 1;
    v12 += v19;
    v20 = v14 == 0;
    v21 = 24;
    v53[v52] = v19;
    if ( !v20 )
      v21 = 36;
    v55[v52] = v17;
    if ( v12 + 1 <= v21 )
      goto LABEL_7;
    goto LABEL_28;
  }
  v37 = 12;
  if ( v52 != 1 )
    v37 = 24;
  if ( v12 + 1 > v37 )
  {
    if ( v52 == 1 )
    {
      v50 = 2;
      v19 = v53[1];
      v38 = 1;
      v18 = 1;
      v17 = v55[1];
LABEL_29:
      v55[v38] = v17;
      v39 = *a1;
      v53[v38] = v19;
      v53[v18] = 0;
      v40 = *(_QWORD *)(v39 + 192);
      v41 = *(_QWORD **)v40;
      if ( *(_QWORD *)v40 )
      {
        *(_QWORD *)v40 = *v41;
      }
      else
      {
        v42 = sub_145CBF0((__int64 *)(v40 + 8), 192, 64);
        v15 = v45;
        v41 = (_QWORD *)v42;
      }
      v44 = 1;
      memset(v41, 0, 0xC0u);
      v55[v18] = v41;
      goto LABEL_8;
    }
    v19 = v53[v47];
    v17 = v55[v47];
    v52 = v49;
    v18 = v47;
    v50 = 2;
LABEL_28:
    v38 = v50++;
    goto LABEL_29;
  }
  v50 = v52;
LABEL_7:
  v44 = 0;
  v52 = 0;
LABEL_8:
  v43 = sub_39461C0(v50, v12, 12, (unsigned int)v53, (unsigned int)v54, v15, 1);
  sub_1F1E940((__int64)v55, v50, v53, (__int64)v54);
  if ( v48 )
    sub_3945E40(v4, a2);
  v46 = 0;
  v22 = 0;
  v23 = a2;
  v24 = a1 + 1;
  for ( i = v23; ; sub_39460A0(v24, i) )
  {
    v26 = v54[v22];
    v27 = v55[v22];
    v28 = (unsigned int)(v26 - 1);
    v29 = *(_QWORD *)(v27 + 8 * v28 + 96);
    if ( v52 != (_DWORD)v22 || !v44 )
      break;
    ++v22;
    v46 = sub_1F1F080(a1, i, v28 | v27 & 0xFFFFFFFFFFFFFFC0LL, v29);
    i += v46;
    if ( v50 == v22 )
      goto LABEL_18;
LABEL_14:
    ;
  }
  *(_DWORD *)(a1[1] + 16LL * i + 8) = v26;
  if ( i )
  {
    v30 = a1[1] + 16LL * (i - 1);
    v31 = (unsigned __int64 *)(*(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 12));
    *v31 = v28 | *v31 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v22;
  sub_1F18EF0((__int64)a1, i, v29);
  if ( v50 != v22 )
    goto LABEL_14;
LABEL_18:
  v32 = i;
  v33 = v24;
  v34 = v32;
  for ( j = v50 - 1; (_DWORD)v43 != j; --j )
    sub_3945E40(v33, v34);
  *(_DWORD *)(a1[1] + 16LL * v34 + 12) = HIDWORD(v43);
  return v46;
}
