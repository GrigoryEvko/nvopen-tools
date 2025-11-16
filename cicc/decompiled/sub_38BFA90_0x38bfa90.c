// Function: sub_38BFA90
// Address: 0x38bfa90
//
__int64 __fastcall sub_38BFA90(
        __int64 a1,
        const void *a2,
        unsigned __int64 a3,
        const void *a4,
        unsigned __int64 a5,
        int a6,
        int a7,
        int a8,
        _BYTE *a9)
{
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // r8
  size_t v15; // r15
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // r10
  __int64 v19; // r15
  unsigned __int8 *v21; // rdi
  __int64 v22; // rax
  unsigned int v23; // r9d
  _QWORD *v24; // rcx
  unsigned __int8 *v25; // r8
  _QWORD *v26; // r10
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // [rsp+8h] [rbp-D8h]
  _QWORD *v31; // [rsp+8h] [rbp-D8h]
  _QWORD *v32; // [rsp+10h] [rbp-D0h]
  _QWORD *v33; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v36; // [rsp+38h] [rbp-A8h]
  unsigned int v37; // [rsp+38h] [rbp-A8h]
  unsigned int v38; // [rsp+38h] [rbp-A8h]
  __int64 v39; // [rsp+38h] [rbp-A8h]
  __int64 v40; // [rsp+38h] [rbp-A8h]
  _BYTE *v41; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v42; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v43; // [rsp+60h] [rbp-80h] BYREF
  __int64 v44; // [rsp+68h] [rbp-78h]
  _BYTE dest[112]; // [rsp+70h] [rbp-70h] BYREF

  v43 = dest;
  v44 = 0x4000000000LL;
  if ( a3 > 0x40 )
  {
    sub_16CD150((__int64)&v43, dest, a3, 1, a5, a6);
    v21 = &v43[(unsigned int)v44];
  }
  else
  {
    if ( !a3 )
    {
      LODWORD(v44) = 0;
      v12 = 0;
      goto LABEL_4;
    }
    v21 = dest;
  }
  memcpy(v21, a2, a3);
  LODWORD(v44) = a3 + v44;
  v12 = (unsigned int)v44;
  if ( HIDWORD(v44) <= (unsigned int)v44 )
  {
    sub_16CD150((__int64)&v43, dest, 0, 1, a5, a6);
    v12 = (unsigned int)v44;
  }
LABEL_4:
  v43[v12] = 44;
  LODWORD(v44) = v44 + 1;
  v13 = (unsigned int)v44;
  if ( HIDWORD(v44) - (unsigned __int64)(unsigned int)v44 < a5 )
  {
    sub_16CD150((__int64)&v43, dest, a5 + (unsigned int)v44, 1, a5, a6);
    v13 = (unsigned int)v44;
  }
  v14 = v43;
  if ( a5 )
  {
    memcpy(&v43[v13], a4, a5);
    v14 = v43;
    LODWORD(v13) = v44;
  }
  v36 = v14;
  v15 = (unsigned int)(a5 + v13);
  LODWORD(v44) = a5 + v13;
  v16 = sub_16D19C0(a1 + 1168, v14, v15);
  v17 = (_QWORD *)(*(_QWORD *)(a1 + 1168) + 8LL * v16);
  v18 = *v17;
  if ( *v17 )
  {
    if ( v18 != -8 )
      goto LABEL_10;
    --*(_DWORD *)(a1 + 1184);
  }
  v30 = v36;
  v32 = v17;
  v37 = v16;
  v22 = malloc(v15 + 17);
  v23 = v37;
  v24 = v32;
  v25 = v30;
  v26 = (_QWORD *)v22;
  if ( !v22 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v26 = 0;
    v25 = v30;
    v24 = v32;
    v23 = v37;
  }
  if ( v15 )
  {
    v31 = v26;
    v33 = v24;
    v38 = v23;
    memcpy(v26 + 2, v25, v15);
    v26 = v31;
    v24 = v33;
    v23 = v38;
  }
  *((_BYTE *)v26 + v15 + 16) = 0;
  *v26 = v15;
  v26[1] = 0;
  *v24 = v26;
  ++*(_DWORD *)(a1 + 1180);
  v27 = (__int64 *)(*(_QWORD *)(a1 + 1168) + 8LL * (unsigned int)sub_16D1CD0(a1 + 1168, v23));
  v18 = *v27;
  if ( *v27 == -8 || !v18 )
  {
    v28 = v27 + 1;
    do
    {
      do
        v18 = *v28++;
      while ( !v18 );
    }
    while ( v18 == -8 );
    v19 = *(_QWORD *)(v18 + 8);
    if ( v19 )
      goto LABEL_11;
    goto LABEL_28;
  }
LABEL_10:
  v19 = *(_QWORD *)(v18 + 8);
  if ( v19 )
    goto LABEL_11;
LABEL_28:
  if ( a9 )
  {
    v42 = 257;
    if ( *a9 )
    {
      v41 = a9;
      LOBYTE(v42) = 3;
    }
    v39 = v18;
    v29 = sub_38BF8E0(a1, (__int64)&v41, 0, 1);
    v18 = v39;
    a9 = (_BYTE *)v29;
  }
  v40 = v18;
  v19 = sub_145CBF0((__int64 *)(a1 + 360), 192, 8);
  sub_38D9FD0(v19, (_DWORD)a2, a3, (_DWORD)a4, a5, a6, a7, a8, (__int64)a9);
  *(_QWORD *)(v40 + 8) = v19;
LABEL_11:
  if ( v43 != dest )
    _libc_free((unsigned __int64)v43);
  return v19;
}
