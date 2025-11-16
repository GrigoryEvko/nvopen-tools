// Function: sub_1218D50
// Address: 0x1218d50
//
__int64 __fastcall sub_1218D50(__int64 a1, __int64 **a2)
{
  unsigned __int64 **v2; // rsi
  unsigned int v3; // r13d
  unsigned __int64 *v4; // r15
  __int64 v5; // r9
  unsigned __int64 *v6; // rax
  const char *v7; // rcx
  unsigned __int64 *v8; // r14
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // rdi
  _BYTE *v12; // r8
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rax
  const char *v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-300h]
  _BYTE *v19; // [rsp+8h] [rbp-2F8h]
  const char *v20; // [rsp+10h] [rbp-2F0h]
  unsigned __int8 v22; // [rsp+2Fh] [rbp-2D1h] BYREF
  _QWORD v23[2]; // [rsp+30h] [rbp-2D0h] BYREF
  _BYTE v24[48]; // [rsp+40h] [rbp-2C0h] BYREF
  const char *v25; // [rsp+70h] [rbp-290h] BYREF
  __int64 i; // [rsp+78h] [rbp-288h]
  _BYTE v27[17]; // [rsp+80h] [rbp-280h] BYREF
  char v28; // [rsp+91h] [rbp-26Fh]
  unsigned __int64 *v29; // [rsp+100h] [rbp-200h] BYREF
  __int64 v30; // [rsp+108h] [rbp-1F8h]
  _BYTE v31[496]; // [rsp+110h] [rbp-1F0h] BYREF

  v2 = &v29;
  v30 = 0x800000000LL;
  v29 = (unsigned __int64 *)v31;
  v23[0] = v24;
  v23[1] = 0xC00000000LL;
  v3 = sub_12186F0(a1, (__int64)&v29, (__int64)v23, &v22);
  if ( (_BYTE)v3 )
    goto LABEL_9;
  v4 = v29;
  v5 = (__int64)&v29[7 * (unsigned int)v30];
  if ( v29 == (unsigned __int64 *)v5 )
  {
    v12 = v27;
    v14 = 0;
    v25 = v27;
    v2 = (unsigned __int64 **)v27;
    i = 0x1000000000LL;
    goto LABEL_26;
  }
  v6 = v29;
  do
  {
    if ( v6[4] )
    {
      v28 = 1;
      v7 = "argument name invalid in function type";
LABEL_8:
      v25 = v7;
      v27[16] = 3;
      v2 = (unsigned __int64 **)*v6;
      v3 = 1;
      sub_11FD800(a1 + 176, *v6, (__int64)&v25, 1);
      goto LABEL_9;
    }
    if ( v6[2] )
    {
      v28 = 1;
      v7 = "argument attributes invalid in function type";
      goto LABEL_8;
    }
    v6 += 7;
  }
  while ( (unsigned __int64 *)v5 != v6 );
  v12 = v27;
  v13 = 16;
  v14 = 0;
  v25 = v27;
  for ( i = 0x1000000000LL; ; v13 = HIDWORD(i) )
  {
    v15 = v4[1];
    if ( v14 + 1 > v13 )
    {
      v18 = v5;
      v19 = v12;
      sub_C8D5F0((__int64)&v25, v12, v14 + 1, 8u, (__int64)v12, v5);
      v14 = (unsigned int)i;
      v5 = v18;
      v12 = v19;
    }
    v4 += 7;
    *(_QWORD *)&v25[8 * v14] = v15;
    v14 = (unsigned int)(i + 1);
    LODWORD(i) = i + 1;
    if ( (unsigned __int64 *)v5 == v4 )
      break;
  }
  v2 = (unsigned __int64 **)v25;
LABEL_26:
  v20 = v12;
  v16 = sub_BCF480(*a2, v2, v14, v22);
  v17 = v25;
  *a2 = (__int64 *)v16;
  if ( v17 != v20 )
    _libc_free(v17, v2);
LABEL_9:
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0], v2);
  v8 = v29;
  v9 = &v29[7 * (unsigned int)v30];
  if ( v29 != v9 )
  {
    do
    {
      v9 -= 7;
      v10 = (unsigned __int64 *)v9[3];
      if ( v10 != v9 + 5 )
      {
        v2 = (unsigned __int64 **)(v9[5] + 1);
        j_j___libc_free_0(v10, v2);
      }
    }
    while ( v8 != v9 );
    v9 = v29;
  }
  if ( v9 != (unsigned __int64 *)v31 )
    _libc_free(v9, v2);
  return v3;
}
