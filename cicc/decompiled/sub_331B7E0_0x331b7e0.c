// Function: sub_331B7E0
// Address: 0x331b7e0
//
__int64 __fastcall sub_331B7E0(__int64 *a1, unsigned int a2)
{
  __int64 v3; // r15
  __int64 *v5; // rbx
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
  unsigned int v23; // eax
  __int64 *v24; // r13
  unsigned int i; // ebx
  int v26; // r9d
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 *v31; // rax
  unsigned int v32; // eax
  unsigned int *v33; // rbx
  unsigned int v34; // r13d
  unsigned int j; // r15d
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // rsi
  unsigned __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-A0h]
  char v46; // [rsp+Fh] [rbp-91h]
  unsigned int v47; // [rsp+10h] [rbp-90h]
  unsigned __int8 v48; // [rsp+10h] [rbp-90h]
  __int64 v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned int v51; // [rsp+28h] [rbp-78h]
  unsigned int v52; // [rsp+28h] [rbp-78h]
  unsigned int v53; // [rsp+2Ch] [rbp-74h]
  unsigned int v54; // [rsp+2Ch] [rbp-74h]
  unsigned int v55[4]; // [rsp+30h] [rbp-70h] BYREF
  _DWORD v56[4]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v57[10]; // [rsp+50h] [rbp-50h] BYREF

  v3 = 16LL * a2;
  v5 = a1 + 1;
  v53 = *(_DWORD *)(a1[1] + v3 + 12);
  v6 = sub_F03A30(a1 + 1, a2);
  v7 = v53;
  v50 = v6;
  if ( v6 )
  {
    v54 = 2;
    v8 = 1;
    v57[0] = v6 & 0xFFFFFFFFFFFFFFC0LL;
    v9 = (v6 & 0x3F) + 1;
    v55[0] = v9;
    v7 += v9;
  }
  else
  {
    v54 = 1;
    v9 = 0;
    v8 = 0;
  }
  v10 = a1[1] + v3;
  v11 = *(_DWORD *)(v10 + 8);
  v47 = v7;
  v51 = v8;
  v55[v8] = v11;
  v12 = v11 + v9;
  v49 = v8;
  v57[v8] = *(_QWORD *)v10;
  v13 = sub_F03C90(v5, a2);
  v14 = v51;
  v15 = v47;
  if ( v13 )
  {
    v16 = v13;
    v17 = v13 & 0xFFFFFFFFFFFFFFC0LL;
    v18 = v54;
    v52 = v51 + 2;
    v19 = (v16 & 0x3F) + 1;
    v12 += v19;
    v20 = v14 == 0;
    v21 = 24;
    v55[v54] = v19;
    if ( !v20 )
      v21 = 36;
    v57[v54] = v17;
    if ( v12 + 1 <= v21 )
      goto LABEL_7;
    goto LABEL_28;
  }
  v37 = 12;
  if ( v54 != 1 )
    v37 = 24;
  if ( v12 + 1 > v37 )
  {
    if ( v54 == 1 )
    {
      v52 = 2;
      v19 = v55[1];
      v38 = 1;
      v18 = 1;
      v17 = v57[1];
      goto LABEL_29;
    }
    v19 = v55[v49];
    v17 = v57[v49];
    v54 = v51;
    v18 = v49;
    v52 = 2;
LABEL_28:
    v38 = v52++;
LABEL_29:
    v57[v38] = v17;
    v39 = *a1;
    v55[v38] = v19;
    v55[v18] = 0;
    v40 = *(_QWORD *)(v39 + 144);
    v41 = *(_QWORD **)v40;
    if ( *(_QWORD *)v40 )
    {
      *(_QWORD *)v40 = *v41;
    }
    else
    {
      v42 = *(_QWORD *)(v40 + 8);
      *(_QWORD *)(v40 + 88) += 192LL;
      v43 = (v42 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      if ( *(_QWORD *)(v40 + 16) >= v43 + 192 && v42 )
      {
        *(_QWORD *)(v40 + 8) = v43 + 192;
        if ( !v43 )
          goto LABEL_32;
        v41 = (_QWORD *)((v42 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      }
      else
      {
        v44 = sub_9D1E70(v40 + 8, 192, 192, 6);
        v15 = v47;
        v41 = (_QWORD *)v44;
      }
    }
    memset(v41, 0, 0xC0u);
LABEL_32:
    v57[v18] = v41;
    v46 = 1;
    goto LABEL_8;
  }
  v52 = v54;
LABEL_7:
  v46 = 0;
  v54 = 0;
LABEL_8:
  v45 = sub_F03E60(v52, v12, 12, (__int64)v55, (__int64)v56, v15, 1u);
  sub_331B460((__int64)v57, v52, v55, (__int64)v56);
  if ( v50 )
    sub_F03AD0((unsigned int *)v5, a2);
  v48 = 0;
  v22 = 0;
  v23 = a2;
  v24 = a1 + 1;
  for ( i = v23; ; sub_F03D40(v24, i) )
  {
    v26 = v56[v22];
    v27 = v57[v22];
    v28 = (unsigned int)(v26 - 1);
    v29 = *(_QWORD *)(v27 + 8 * v28 + 96);
    if ( v54 == (_DWORD)v22 && v46 )
    {
      v48 = sub_331BBE0(a1, i, v28 | v27 & 0xFFFFFFFFFFFFFFC0LL, v29);
      i += v48;
      goto LABEL_14;
    }
    *(_DWORD *)(a1[1] + 16LL * i + 8) = v26;
    if ( i )
      break;
LABEL_14:
    if ( v52 == ++v22 )
      goto LABEL_18;
LABEL_15:
    ;
  }
  ++v22;
  v30 = a1[1] + 16LL * (i - 1);
  v31 = (unsigned __int64 *)(*(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 12));
  *v31 = v28 | *v31 & 0xFFFFFFFFFFFFFFC0LL;
  sub_325DE80((__int64)a1, i, v29);
  if ( v52 != v22 )
    goto LABEL_15;
LABEL_18:
  v32 = i;
  v33 = (unsigned int *)v24;
  v34 = v32;
  for ( j = v52 - 1; j != (_DWORD)v45; --j )
    sub_F03AD0(v33, v34);
  *(_DWORD *)(a1[1] + 16LL * v34 + 12) = HIDWORD(v45);
  return v48;
}
