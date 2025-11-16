// Function: sub_20FE530
// Address: 0x20fe530
//
__int64 __fastcall sub_20FE530(_QWORD *a1, unsigned int a2)
{
  _QWORD *v2; // r13
  __int64 v4; // rbx
  int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned int v8; // ecx
  unsigned int v9; // r10d
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r8
  unsigned int v14; // r10d
  char v15; // dl
  unsigned __int64 v16; // rax
  int v17; // edx
  unsigned int v18; // ecx
  __int64 v19; // r15
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int64 *v25; // rax
  unsigned int i; // r14d
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // [rsp+10h] [rbp-A0h]
  char v33; // [rsp+1Eh] [rbp-92h]
  unsigned __int8 v34; // [rsp+1Fh] [rbp-91h]
  unsigned int v35; // [rsp+20h] [rbp-90h]
  unsigned int v36; // [rsp+20h] [rbp-90h]
  unsigned int v37; // [rsp+28h] [rbp-88h]
  __int64 v38; // [rsp+28h] [rbp-88h]
  __int64 v39; // [rsp+30h] [rbp-80h]
  unsigned int v40; // [rsp+30h] [rbp-80h]
  unsigned int v41; // [rsp+38h] [rbp-78h]
  unsigned int v42[4]; // [rsp+40h] [rbp-70h] BYREF
  _DWORD v43[4]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v44[10]; // [rsp+60h] [rbp-50h] BYREF

  v2 = a1 + 1;
  v4 = a2;
  v5 = *(_DWORD *)(a1[1] + 16LL * a2 + 12);
  v6 = sub_3945DA0(a1 + 1, a2);
  v7 = v6;
  if ( v6 )
  {
    v41 = 2;
    v8 = 1;
    v44[0] = v6 & 0xFFFFFFFFFFFFFFC0LL;
    v9 = (v6 & 0x3F) + 1;
    v42[0] = v9;
    v5 += v9;
  }
  else
  {
    v41 = 1;
    v9 = 0;
    v8 = 0;
  }
  v10 = a1[1] + 16LL * a2;
  v11 = *(_DWORD *)(v10 + 8);
  v35 = v8;
  v39 = v8;
  v42[v8] = v11;
  v37 = v11 + v9;
  v44[v8] = *(_QWORD *)v10;
  v12 = sub_3945FF0(v2, a2);
  v13 = v39;
  v14 = v37;
  if ( v12 )
  {
    v15 = v12;
    v16 = v12 & 0xFFFFFFFFFFFFFFC0LL;
    v13 = v41;
    v40 = v35 + 2;
    v17 = (v15 & 0x3F) + 1;
    v14 = v17 + v37;
    v18 = 24;
    v42[v41] = v17;
    if ( v35 )
      v18 = 36;
    v44[v41] = v16;
    if ( v14 + 1 <= v18 )
      goto LABEL_7;
    goto LABEL_28;
  }
  v28 = 12;
  if ( v41 != 1 )
    v28 = 24;
  if ( v37 + 1 > v28 )
  {
    if ( v41 == 1 )
    {
      v40 = 2;
      v17 = v42[1];
      v29 = 1;
      v13 = 1;
      v16 = v44[1];
LABEL_29:
      v44[v29] = v16;
      v36 = v14;
      v30 = *(_QWORD *)(*a1 + 200LL);
      v42[v29] = v17;
      v38 = v13;
      v42[v13] = 0;
      v31 = sub_20FC1F0(v30);
      v33 = 1;
      v14 = v36;
      v44[v38] = v31;
      goto LABEL_8;
    }
    v17 = v42[v39];
    v16 = v44[v39];
    v41 = v35;
    v40 = 2;
LABEL_28:
    v29 = v40++;
    goto LABEL_29;
  }
  v40 = v41;
LABEL_7:
  v33 = 0;
  v41 = 0;
LABEL_8:
  v32 = sub_39461C0(v40, v14, 12, (unsigned int)v42, (unsigned int)v43, v5, 1);
  sub_20FE1B0((__int64)v44, v40, v42, (__int64)v43);
  if ( v7 )
    sub_3945E40(v2, a2);
  v34 = 0;
  v19 = 0;
  while ( 1 )
  {
    v20 = v43[v19];
    v21 = v44[v19];
    v22 = (unsigned int)(v20 - 1);
    v23 = *(_QWORD *)(v21 + 8 * v22 + 96);
    if ( v41 != (_DWORD)v19 || !v33 )
      break;
    ++v19;
    v34 = sub_20FE8C0(a1, (unsigned int)v4, v22 | v21 & 0xFFFFFFFFFFFFFFC0LL, v23);
    v4 = v34 + (unsigned int)v4;
    if ( v40 == v19 )
      goto LABEL_18;
LABEL_14:
    sub_39460A0(v2, (unsigned int)v4);
  }
  *(_DWORD *)(a1[1] + 16LL * (unsigned int)v4 + 8) = v20;
  if ( (_DWORD)v4 )
  {
    v24 = a1[1] + 16LL * (unsigned int)(v4 - 1);
    v25 = (unsigned __int64 *)(*(_QWORD *)v24 + 8LL * *(unsigned int *)(v24 + 12));
    *v25 = v22 | *v25 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v19;
  sub_20FCF40((__int64)a1, v4, v23);
  if ( v40 != v19 )
    goto LABEL_14;
LABEL_18:
  for ( i = v40 - 1; (_DWORD)v32 != i; --i )
    sub_3945E40(v2, (unsigned int)v4);
  *(_DWORD *)(a1[1] + 16 * v4 + 12) = HIDWORD(v32);
  return v34;
}
