// Function: sub_2E1C2B0
// Address: 0x2e1c2b0
//
__int64 __fastcall sub_2E1C2B0(__int64 *a1, unsigned int a2)
{
  __int64 v3; // r15
  __int64 *v4; // r12
  __int64 v6; // rax
  unsigned int v7; // r9d
  unsigned int v8; // edx
  unsigned int v9; // r14d
  __int64 v10; // r15
  unsigned int v11; // eax
  int v12; // r14d
  __int64 v13; // rax
  unsigned int v14; // edx
  unsigned int v15; // r9d
  char v16; // si
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  int v19; // esi
  bool v20; // zf
  unsigned int v21; // edx
  __int64 v22; // r14
  unsigned int i; // r15d
  int v24; // r9d
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rax
  unsigned __int64 *v29; // rax
  unsigned int v30; // r13d
  unsigned int j; // r15d
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rdx
  __int64 v38; // rsi
  unsigned __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-A0h]
  char v42; // [rsp+Fh] [rbp-91h]
  unsigned int v43; // [rsp+10h] [rbp-90h]
  unsigned __int8 v44; // [rsp+10h] [rbp-90h]
  __int64 v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+20h] [rbp-80h]
  unsigned int v47; // [rsp+28h] [rbp-78h]
  unsigned int v48; // [rsp+28h] [rbp-78h]
  unsigned int v49; // [rsp+2Ch] [rbp-74h]
  unsigned int v50; // [rsp+2Ch] [rbp-74h]
  unsigned int v51[4]; // [rsp+30h] [rbp-70h] BYREF
  _DWORD v52[4]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v53[10]; // [rsp+50h] [rbp-50h] BYREF

  v3 = 16LL * a2;
  v4 = a1 + 1;
  v49 = *(_DWORD *)(a1[1] + v3 + 12);
  v6 = sub_F03A30(a1 + 1, a2);
  v7 = v49;
  v46 = v6;
  if ( v6 )
  {
    v50 = 2;
    v8 = 1;
    v53[0] = v6 & 0xFFFFFFFFFFFFFFC0LL;
    v9 = (v6 & 0x3F) + 1;
    v51[0] = v9;
    v7 += v9;
  }
  else
  {
    v50 = 1;
    v9 = 0;
    v8 = 0;
  }
  v10 = a1[1] + v3;
  v11 = *(_DWORD *)(v10 + 8);
  v43 = v7;
  v47 = v8;
  v51[v8] = v11;
  v12 = v11 + v9;
  v45 = v8;
  v53[v8] = *(_QWORD *)v10;
  v13 = sub_F03C90(v4, a2);
  v14 = v47;
  v15 = v43;
  if ( v13 )
  {
    v16 = v13;
    v17 = v13 & 0xFFFFFFFFFFFFFFC0LL;
    v18 = v50;
    v48 = v47 + 2;
    v19 = (v16 & 0x3F) + 1;
    v12 += v19;
    v20 = v14 == 0;
    v21 = 24;
    v51[v50] = v19;
    if ( !v20 )
      v21 = 36;
    v53[v50] = v17;
    if ( v12 + 1 <= v21 )
      goto LABEL_7;
    goto LABEL_28;
  }
  v33 = 12;
  if ( v50 != 1 )
    v33 = 24;
  if ( v12 + 1 > v33 )
  {
    if ( v50 == 1 )
    {
      v48 = 2;
      v19 = v51[1];
      v34 = 1;
      v18 = 1;
      v17 = v53[1];
      goto LABEL_29;
    }
    v19 = v51[v45];
    v17 = v53[v45];
    v50 = v47;
    v18 = v45;
    v48 = 2;
LABEL_28:
    v34 = v48++;
LABEL_29:
    v53[v34] = v17;
    v35 = *a1;
    v51[v34] = v19;
    v51[v18] = 0;
    v36 = *(_QWORD *)(v35 + 200);
    v37 = *(_QWORD **)v36;
    if ( *(_QWORD *)v36 )
    {
      *(_QWORD *)v36 = *v37;
    }
    else
    {
      v38 = *(_QWORD *)(v36 + 8);
      *(_QWORD *)(v36 + 88) += 192LL;
      v39 = (v38 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_QWORD *)(v36 + 16) >= v39 + 192 && v38 )
      {
        *(_QWORD *)(v36 + 8) = v39 + 192;
        if ( !v39 )
          goto LABEL_32;
        v37 = (_QWORD *)((v38 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      }
      else
      {
        v40 = sub_9D1E70(v36 + 8, 192, 192, 6);
        v15 = v43;
        v37 = (_QWORD *)v40;
      }
    }
    memset(v37, 0, 0xC0u);
    memset(
      (void *)((unsigned __int64)(v37 + 13) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v37 - (((_DWORD)v37 + 104) & 0xFFFFFFF8) + 192) >> 3));
LABEL_32:
    v53[v18] = v37;
    v42 = 1;
    goto LABEL_8;
  }
  v48 = v50;
LABEL_7:
  v42 = 0;
  v50 = 0;
LABEL_8:
  v41 = sub_F03E60(v48, v12, 12, (__int64)v51, (__int64)v52, v15, 1u);
  sub_2E1BF30((__int64)v53, v48, v51, (__int64)v52);
  if ( v46 )
    sub_F03AD0((unsigned int *)v4, a2);
  v44 = 0;
  v22 = 0;
  for ( i = a2; ; sub_F03D40(v4, i) )
  {
    v24 = v52[v22];
    v25 = v53[v22];
    v26 = (unsigned int)(v24 - 1);
    v27 = *(_QWORD *)(v25 + 8 * v26 + 96);
    if ( v50 != (_DWORD)v22 || !v42 )
      break;
    ++v22;
    v44 = sub_2E1C6D0(a1, i, v26 | v25 & 0xFFFFFFFFFFFFFFC0LL, v27);
    i += v44;
    if ( v48 == v22 )
      goto LABEL_18;
LABEL_14:
    ;
  }
  *(_DWORD *)(a1[1] + 16LL * i + 8) = v24;
  if ( i )
  {
    v28 = a1[1] + 16LL * (i - 1);
    v29 = (unsigned __int64 *)(*(_QWORD *)v28 + 8LL * *(unsigned int *)(v28 + 12));
    *v29 = v26 | *v29 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v22;
  sub_2E1A5E0((__int64)a1, i, v27);
  if ( v48 != v22 )
    goto LABEL_14;
LABEL_18:
  v30 = i;
  for ( j = v48 - 1; j != (_DWORD)v41; --j )
    sub_F03AD0((unsigned int *)v4, v30);
  *(_DWORD *)(a1[1] + 16LL * v30 + 12) = HIDWORD(v41);
  return v44;
}
