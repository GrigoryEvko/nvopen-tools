// Function: sub_14AC030
// Address: 0x14ac030
//
__int64 __fastcall sub_14AC030(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  signed __int64 v5; // r8
  __int64 v6; // rbx
  unsigned int *v7; // r12
  __int64 v8; // r15
  unsigned int *v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 result; // rax
  unsigned int v13; // edx
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // ebx
  const void *v17; // r9
  size_t v18; // r11
  __int64 v19; // rax
  __int64 v20; // rbx
  _BYTE *v21; // rsi
  unsigned __int64 v22; // rdi
  _DWORD *v23; // rax
  _DWORD *v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  size_t v27; // r8
  __int64 v28; // r11
  size_t v29; // r8
  _BYTE *v30; // rdi
  size_t n; // [rsp+0h] [rbp-90h]
  __int64 v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  unsigned __int64 v34; // [rsp+10h] [rbp-80h]
  const void *v35; // [rsp+10h] [rbp-80h]
  signed __int64 v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  signed __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  signed __int64 v41; // [rsp+18h] [rbp-78h]
  _BYTE *v42; // [rsp+20h] [rbp-70h] BYREF
  __int64 v43; // [rsp+28h] [rbp-68h]
  _BYTE v44[96]; // [rsp+30h] [rbp-60h] BYREF

LABEL_1:
  v4 = a1;
  if ( !a3 )
    return a1;
  v5 = 4 * a3;
  v6 = a3;
  v7 = a2;
  v8 = a4;
  v9 = &a2[a3];
  while ( 1 )
  {
    v10 = *(_BYTE *)(v4 + 16);
    if ( v10 <= 0x10u )
    {
      v11 = sub_15A0A60(v4, *v7);
      a2 = v7 + 1;
      a3 = v6 - 1;
      a1 = v11;
      if ( v11 )
      {
LABEL_5:
        a4 = v8;
        goto LABEL_1;
      }
      return 0;
    }
    if ( v10 != 87 )
    {
      if ( v10 != 86 )
        return 0;
      v13 = *(_DWORD *)(v4 + 64);
      v43 = 0x500000000LL;
      v14 = 5;
      v15 = 0;
      v42 = v44;
      if ( v13 + (unsigned int)v6 > 5uLL )
      {
        v36 = v5;
        sub_16CD150(&v42, v44, v13 + (unsigned int)v6, 4);
        v15 = (unsigned int)v43;
        v13 = *(_DWORD *)(v4 + 64);
        v5 = v36;
        v14 = HIDWORD(v43) - (unsigned __int64)(unsigned int)v43;
      }
      v16 = v13;
      v17 = *(const void **)(v4 + 56);
      v18 = 4LL * v13;
      if ( v13 > v14 )
      {
        n = v5;
        v32 = 4LL * v13;
        v35 = *(const void **)(v4 + 56);
        sub_16CD150(&v42, v44, v13 + v15, 4);
        v15 = (unsigned int)v43;
        v5 = n;
        v18 = v32;
        v17 = v35;
      }
      if ( v18 )
      {
        v38 = v5;
        memcpy(&v42[4 * v15], v17, v18);
        LODWORD(v15) = v43;
        v5 = v38;
      }
      LODWORD(v43) = v16 + v15;
      v19 = v16 + (unsigned int)v15;
      v20 = v5 >> 2;
      if ( v5 >> 2 > HIDWORD(v43) - (unsigned __int64)(unsigned int)v19 )
      {
        v41 = v5;
        sub_16CD150(&v42, v44, v19 + v20, 4);
        v19 = (unsigned int)v43;
        v5 = v41;
      }
      v21 = v42;
      if ( v5 )
      {
        memcpy(&v42[4 * v19], v7, v5);
        v21 = v42;
        LODWORD(v19) = v43;
      }
      LODWORD(v43) = v20 + v19;
      result = sub_14AC030(*(_QWORD *)(v4 - 24), v21, (unsigned int)(v20 + v19), v8);
      v22 = (unsigned __int64)v42;
      if ( v42 == v44 )
        return result;
LABEL_21:
      v39 = result;
      _libc_free(v22);
      return v39;
    }
    v23 = *(_DWORD **)(v4 + 56);
    v24 = &v23[*(unsigned int *)(v4 + 64)];
    if ( v23 == v24 )
    {
      a2 = v7;
LABEL_31:
      a1 = *(_QWORD *)(v4 - 24);
      a3 = v9 - a2;
      goto LABEL_5;
    }
    if ( v7 == v9 )
      break;
    a2 = v7;
    while ( *a2 == *v23 )
    {
      ++v23;
      ++a2;
      if ( v24 == v23 )
        goto LABEL_31;
      if ( a2 == v9 )
        goto LABEL_32;
    }
    v4 = *(_QWORD *)(v4 - 48);
  }
LABEL_32:
  if ( !v8 )
    return 0;
  v34 = v5;
  v25 = v5 >> 2;
  v40 = sub_15FB2A0(*(_QWORD *)v4, v7, v5 >> 2);
  v26 = sub_1599EF0(v40);
  v27 = v34;
  v42 = v44;
  v28 = v26;
  v43 = 0xA00000000LL;
  if ( v34 > 0x28 )
  {
    v33 = v26;
    sub_16CD150(&v42, v44, v25, 4);
    v28 = v33;
    v27 = v34;
    v30 = &v42[4 * (unsigned int)v43];
  }
  else
  {
    if ( !v34 )
      goto LABEL_35;
    v30 = v44;
  }
  v37 = v28;
  memcpy(v30, v7, v27);
  v27 = (unsigned int)v43;
  v28 = v37;
LABEL_35:
  v29 = v25 + v27;
  LODWORD(v43) = v29;
  result = sub_14AC3D0(v4, v28, v40, &v42, v29, v8);
  v22 = (unsigned __int64)v42;
  if ( v42 != v44 )
    goto LABEL_21;
  return result;
}
