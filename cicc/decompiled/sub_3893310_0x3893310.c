// Function: sub_3893310
// Address: 0x3893310
//
__int64 __fastcall sub_3893310(__int64 a1, __int64 **a2)
{
  int v3; // r9d
  unsigned int v4; // r13d
  unsigned __int64 *v5; // rcx
  unsigned __int64 *v6; // rax
  __int64 v7; // r8
  unsigned __int64 v8; // r12
  const char *v9; // rcx
  unsigned __int64 *v10; // r14
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  unsigned int v14; // esi
  __int64 v15; // rdx
  char *v16; // rax
  char *v17; // rsi
  __int64 v18; // rax
  char *v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-2B8h]
  char *v21; // [rsp+10h] [rbp-2B0h]
  unsigned __int8 v22; // [rsp+2Fh] [rbp-291h] BYREF
  char *v23; // [rsp+30h] [rbp-290h] BYREF
  __int64 v24; // [rsp+38h] [rbp-288h]
  char v25; // [rsp+40h] [rbp-280h] BYREF
  char v26; // [rsp+41h] [rbp-27Fh]
  unsigned __int64 *v27; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v28; // [rsp+C8h] [rbp-1F8h]
  _BYTE v29[496]; // [rsp+D0h] [rbp-1F0h] BYREF

  v27 = (unsigned __int64 *)v29;
  v28 = 0x800000000LL;
  v4 = sub_3892DD0(a1, (__int64)&v27, &v22);
  if ( (_BYTE)v4 )
    goto LABEL_9;
  if ( !(_DWORD)v28 )
  {
    v15 = 0;
    v23 = &v25;
    v17 = &v25;
    v24 = 0x1000000000LL;
    goto LABEL_24;
  }
  v5 = v27;
  v6 = v27;
  v7 = 56LL * (unsigned int)(v28 - 1);
  do
  {
    v8 = v6[4];
    if ( v8 )
    {
      v26 = 1;
      v9 = "argument name invalid in function type";
LABEL_8:
      v23 = (char *)v9;
      v25 = 3;
      v4 = sub_38814C0(a1 + 8, *v6, (__int64)&v23);
      goto LABEL_9;
    }
    if ( v6[2] )
    {
      v26 = 1;
      v9 = "argument attributes invalid in function type";
      goto LABEL_8;
    }
    v6 += 7;
  }
  while ( (unsigned __int64 *)((char *)v27 + v7 + 56) != v6 );
  v14 = 16;
  v15 = 0;
  v23 = &v25;
  v24 = 0x1000000000LL;
  while ( 1 )
  {
    v16 = (char *)v5 + v8;
    if ( (unsigned int)v15 >= v14 )
    {
      v20 = v7;
      v21 = (char *)v5 + v8;
      sub_16CD150((__int64)&v23, &v25, 0, 8, v7, v3);
      v15 = (unsigned int)v24;
      v7 = v20;
      v16 = v21;
    }
    *(_QWORD *)&v23[8 * v15] = *((_QWORD *)v16 + 1);
    v15 = (unsigned int)(v24 + 1);
    LODWORD(v24) = v24 + 1;
    if ( v7 == v8 )
      break;
    v5 = v27;
    v14 = HIDWORD(v24);
    v8 += 56LL;
  }
  v17 = v23;
LABEL_24:
  v18 = sub_1644EA0(*a2, v17, v15, v22);
  v19 = v23;
  *a2 = (__int64 *)v18;
  if ( v19 != &v25 )
    _libc_free((unsigned __int64)v19);
LABEL_9:
  v10 = v27;
  v11 = (unsigned __int64)&v27[7 * (unsigned int)v28];
  if ( v27 != (unsigned __int64 *)v11 )
  {
    do
    {
      v11 -= 56LL;
      v12 = *(_QWORD *)(v11 + 24);
      if ( v12 != v11 + 40 )
        j_j___libc_free_0(v12);
    }
    while ( v10 != (unsigned __int64 *)v11 );
    v11 = (unsigned __int64)v27;
  }
  if ( (_BYTE *)v11 != v29 )
    _libc_free(v11);
  return v4;
}
