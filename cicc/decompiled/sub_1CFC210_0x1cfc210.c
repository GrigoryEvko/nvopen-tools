// Function: sub_1CFC210
// Address: 0x1cfc210
//
__int64 __fastcall sub_1CFC210(_QWORD *a1, unsigned __int64 **a2)
{
  __int64 v2; // rdx
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // rbx
  unsigned int v6; // ebx
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rsi
  __int64 v14; // r8
  _QWORD *v15; // rcx
  _QWORD *v16; // r15
  unsigned __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rdi
  int v24; // esi
  int v25; // r9d
  unsigned int v27; // [rsp+Ch] [rbp-B4h]
  __int64 v28; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v29; // [rsp+18h] [rbp-A8h]
  _QWORD *v30; // [rsp+28h] [rbp-98h]
  __int64 v31; // [rsp+30h] [rbp-90h] BYREF
  __int64 v32; // [rsp+38h] [rbp-88h]
  __int64 v33; // [rsp+40h] [rbp-80h]
  int v34; // [rsp+48h] [rbp-78h]
  _BYTE v35[40]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v36; // [rsp+78h] [rbp-48h]
  unsigned __int64 *v37; // [rsp+80h] [rbp-40h]

  sub_1FE79E0(v35, a1[77], *a2);
  v2 = a1[83];
  v3 = v36;
  v31 = 0;
  v4 = a1[84];
  v33 = 0;
  v32 = 0;
  v34 = 0;
  v5 = (v4 - v2) >> 3;
  if ( !(_DWORD)v5 )
  {
    v22 = 0;
    goto LABEL_20;
  }
  v6 = v5 - 1;
  v28 = v36 + 16;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v2 + 8LL * v6);
    if ( *(__int16 *)(v8 + 24) >= 0 )
    {
      sub_1FEA180(v35, v8, 0, 0, &v31);
      if ( (*(_BYTE *)(v8 + 26) & 1) == 0 )
        goto LABEL_4;
    }
    else
    {
      sub_1FEABF0(v35, v8, 0, 0, &v31);
      if ( (*(_BYTE *)(v8 + 26) & 1) == 0 )
        goto LABEL_4;
    }
    v9 = *(_QWORD *)(a1[78] + 648LL);
    v10 = *(unsigned int *)(v9 + 720);
    if ( (_DWORD)v10 )
      break;
LABEL_4:
    if ( v6-- == 0 )
      goto LABEL_19;
LABEL_5:
    v2 = a1[83];
  }
  v11 = *(_QWORD *)(v9 + 704);
  v12 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = (__int64 *)(v11 + 40LL * v12);
  v14 = *v13;
  if ( v8 != *v13 )
  {
    v24 = 1;
    while ( v14 != -8 )
    {
      v25 = v24 + 1;
      v12 = (v10 - 1) & (v24 + v12);
      v13 = (__int64 *)(v11 + 40LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_10;
      v24 = v25;
    }
    goto LABEL_4;
  }
LABEL_10:
  if ( v13 == (__int64 *)(v11 + 40 * v10) )
    goto LABEL_4;
  v15 = (_QWORD *)v13[1];
  v30 = &v15[*((unsigned int *)v13 + 4)];
  if ( v30 == v15 )
    goto LABEL_4;
  v27 = v6;
  v16 = (_QWORD *)v13[1];
  v17 = v37;
  do
  {
    v18 = *v16;
    if ( !*(_BYTE *)(*v16 + 49LL) )
    {
      v19 = sub_1FE7480(v35, *v16, &v31);
      if ( v19 )
      {
        v29 = v19;
        sub_1DD5BA0(v28, v19);
        v20 = *(_QWORD *)v29;
        v21 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v29 + 8) = v17;
        *(_QWORD *)v29 = v21 | v20 & 7;
        *(_QWORD *)(v21 + 8) = v29;
        *v17 = *v17 & 7 | v29;
      }
      *(_BYTE *)(v18 + 49) = 1;
    }
    ++v16;
  }
  while ( v30 != v16 );
  v6 = v27 - 1;
  if ( v27 )
    goto LABEL_5;
LABEL_19:
  v22 = v32;
  v3 = v36;
LABEL_20:
  *a2 = v37;
  j___libc_free_0(v22);
  return v3;
}
