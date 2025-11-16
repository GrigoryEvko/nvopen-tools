// Function: sub_1A29B80
// Address: 0x1a29b80
//
__int64 __fastcall sub_1A29B80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned int v7; // edx
  _QWORD *v8; // r8
  __int64 v9; // rax
  char v10; // si
  __int64 v11; // rbx
  int v12; // esi
  __int64 v13; // rcx
  int v14; // edi
  unsigned int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rdi
  unsigned __int8 v19; // al
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r15
  __int64 v24; // rbx
  unsigned int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rbx
  unsigned int v29; // r12d
  _QWORD *v30; // rax
  __int64 v32; // rax
  __int64 *v33; // r12
  int v34; // r9d
  char v35; // dl
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  int v40; // r11d
  __int64 v41; // [rsp+18h] [rbp-D8h]
  __int64 v42; // [rsp+18h] [rbp-D8h]
  __int64 v44; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v45; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD *v46; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-A8h]
  _QWORD v48[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v50; // [rsp+78h] [rbp-78h]
  _BYTE *v51; // [rsp+80h] [rbp-70h]
  __int64 v52; // [rsp+88h] [rbp-68h]
  int v53; // [rsp+90h] [rbp-60h]
  _BYTE v54[88]; // [rsp+98h] [rbp-58h] BYREF

  v4 = 0;
  v44 = a2;
  if ( !a1 )
    return v4;
  v49 = 0;
  v50 = v54;
  v51 = v54;
  v46 = v48;
  v52 = 4;
  v53 = 0;
  v47 = 0x400000001LL;
  sub_1412190((__int64)&v49, a2);
  v48[0] = v44;
  v7 = 1;
  while ( 1 )
  {
    v8 = v46;
    v9 = v7;
    v10 = *(_BYTE *)(a4 + 8);
    --v7;
    v11 = v46[v9 - 1];
    LODWORD(v47) = v7;
    v45 = v11;
    v12 = v10 & 1;
    if ( v12 )
    {
      v13 = a4 + 16;
      v14 = 3;
    }
    else
    {
      v21 = *(unsigned int *)(a4 + 24);
      v13 = *(_QWORD *)(a4 + 16);
      if ( !(_DWORD)v21 )
        goto LABEL_53;
      v14 = v21 - 1;
    }
    v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = v13 + 16LL * v15;
    v17 = *(_QWORD *)v16;
    if ( v11 == *(_QWORD *)v16 )
      goto LABEL_6;
    v39 = 1;
    while ( v17 != -8 )
    {
      v40 = v39 + 1;
      v15 = v14 & (v39 + v15);
      v16 = v13 + 16LL * v15;
      v17 = *(_QWORD *)v16;
      if ( v11 == *(_QWORD *)v16 )
        goto LABEL_6;
      v39 = v40;
    }
    if ( (_BYTE)v12 )
    {
      v38 = 64;
      goto LABEL_54;
    }
    v21 = *(unsigned int *)(a4 + 24);
LABEL_53:
    v38 = 16 * v21;
LABEL_54:
    v16 = v13 + v38;
LABEL_6:
    v18 = 64;
    if ( !(_BYTE)v12 )
      v18 = 16LL * *(unsigned int *)(a4 + 24);
    if ( v16 != v18 + v13 && !*(_BYTE *)(v16 + 8) )
      goto LABEL_31;
    v19 = *(_BYTE *)(v11 + 16);
    if ( v19 > 0x17u )
      break;
    if ( v19 > 0x11u )
      goto LABEL_30;
LABEL_15:
    if ( !v7 )
    {
      v4 = 1;
      v20 = sub_1A29A10(a4, &v44);
      v8 = v46;
      *(_BYTE *)(v20 + 8) = 1;
      goto LABEL_32;
    }
  }
  if ( a3 == *(_QWORD *)(v11 + 40) )
  {
    if ( (unsigned __int8)sub_15F2ED0(v11) )
      goto LABEL_30;
    if ( *(_BYTE *)(v11 + 16) != 77 )
    {
      v32 = 3LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
      {
        v33 = *(__int64 **)(v11 - 8);
        v11 = (__int64)&v33[v32];
      }
      else
      {
        v33 = (__int64 *)(v11 - v32 * 8);
      }
      for ( ; v33 != (__int64 *)v11; v33 += 3 )
      {
        sub_1412190((__int64)&v49, *v33);
        if ( v35 )
        {
          v36 = *v33;
          v37 = (unsigned int)v47;
          if ( (unsigned int)v47 >= HIDWORD(v47) )
          {
            v42 = *v33;
            sub_16CD150((__int64)&v46, v48, 0, 8, v36, v34);
            v37 = (unsigned int)v47;
            v36 = v42;
          }
          v46[v37] = v36;
          LODWORD(v47) = v47 + 1;
        }
      }
    }
LABEL_14:
    v7 = v47;
    goto LABEL_15;
  }
  v22 = *(_QWORD *)(a3 + 8);
  if ( !v22 )
    goto LABEL_15;
  v41 = v11;
  v23 = a4;
  v24 = *(_QWORD *)(a3 + 8);
  v25 = v7;
  do
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v24) + 16) - 25) <= 9u )
      break;
    v24 = *(_QWORD *)(v24 + 8);
  }
  while ( v24 );
  v26 = v23;
  v27 = v24;
  v28 = v22;
  v29 = v25;
  a4 = v26;
  do
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v28) + 16) - 25) <= 9u )
      break;
    v28 = *(_QWORD *)(v28 + 8);
  }
  while ( v28 );
  v7 = v29;
  if ( v28 == v27 )
    goto LABEL_15;
  while ( 1 )
  {
    v30 = sub_1648700(v27);
    if ( !sub_15CCE20(a1, v41, v30[5]) )
      break;
    do
      v27 = *(_QWORD *)(v27 + 8);
    while ( v27 && (unsigned __int8)(*((_BYTE *)sub_1648700(v27) + 16) - 25) > 9u );
    if ( v28 == v27 )
      goto LABEL_14;
  }
LABEL_30:
  *(_BYTE *)(sub_1A29A10(a4, &v45) + 8) = 0;
  v8 = v46;
LABEL_31:
  v4 = 0;
LABEL_32:
  if ( v8 != v48 )
    _libc_free((unsigned __int64)v8);
  if ( v51 != v50 )
    _libc_free((unsigned __int64)v51);
  return v4;
}
