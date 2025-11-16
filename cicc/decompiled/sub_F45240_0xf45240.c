// Function: sub_F45240
// Address: 0xf45240
//
__int64 __fastcall sub_F45240(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  _BYTE *v12; // r12
  int v13; // ecx
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  _BYTE *v17; // r10
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // r8
  __int64 v21; // r12
  __int16 v23; // dx
  int v24; // eax
  unsigned __int8 v25; // [rsp+0h] [rbp-90h]
  __int64 v26; // [rsp+0h] [rbp-90h]
  __int64 *v27; // [rsp+10h] [rbp-80h] BYREF
  __int64 v28; // [rsp+18h] [rbp-78h]
  _BYTE v29[112]; // [rsp+20h] [rbp-70h] BYREF

  v28 = 0x800000000LL;
  v6 = *((_BYTE *)a2 - 16);
  v27 = (__int64 *)v29;
  if ( (v6 & 2) == 0 )
  {
    v23 = *((_WORD *)a2 - 8) >> 6;
    a2 -= (v6 >> 2) & 0xF;
    v7 = a2 - 2;
    v8 = (__int64)&a2[(v23 & 0xF) - 2];
    if ( a2 - 2 != (__int64 *)v8 )
      goto LABEL_3;
    return 0;
  }
  v7 = (__int64 *)*(a2 - 4);
  v8 = (__int64)&v7[*((unsigned int *)a2 - 6)];
  if ( v7 == (__int64 *)v8 )
    return 0;
LABEL_3:
  v9 = 0;
  do
  {
    while ( 1 )
    {
      v12 = (_BYTE *)*v7;
      if ( (unsigned __int8)(*(_BYTE *)*v7 - 5) <= 0x1Fu )
        break;
LABEL_7:
      if ( (__int64 *)v8 == ++v7 )
        goto LABEL_15;
    }
    v13 = *((_DWORD *)*a1 + 6);
    a2 = (__int64 *)(*a1)[1];
    if ( !v13 )
      goto LABEL_4;
    v14 = v13 - 1;
    v15 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v16 = &a2[2 * v15];
    v17 = (_BYTE *)*v16;
    if ( v12 != (_BYTE *)*v16 )
    {
      v24 = 1;
      while ( v17 != (_BYTE *)-4096LL )
      {
        a6 = (unsigned int)(v24 + 1);
        v15 = v14 & (v24 + v15);
        v16 = &a2[2 * v15];
        v17 = (_BYTE *)*v16;
        if ( v12 == (_BYTE *)*v16 )
          goto LABEL_11;
        v24 = a6;
      }
LABEL_4:
      v10 = (unsigned int)v28;
      v11 = (unsigned int)v28 + 1LL;
      if ( v11 > HIDWORD(v28) )
      {
        a2 = (__int64 *)v29;
        v25 = v9;
        sub_C8D5F0((__int64)&v27, v29, v11, 8u, v9, a6);
        v10 = (unsigned int)v28;
        v9 = v25;
      }
      v27[v10] = (__int64)v12;
      LODWORD(v28) = v28 + 1;
      goto LABEL_7;
    }
LABEL_11:
    v18 = v16[1];
    if ( !v18 )
      goto LABEL_4;
    v19 = (unsigned int)v28;
    v20 = (unsigned int)v28 + 1LL;
    if ( HIDWORD(v28) < v20 )
    {
      a2 = (__int64 *)v29;
      v26 = v18;
      sub_C8D5F0((__int64)&v27, v29, (unsigned int)v28 + 1LL, 8u, v20, a6);
      v19 = (unsigned int)v28;
      v18 = v26;
    }
    ++v7;
    v9 = 1;
    v27[v19] = v18;
    LODWORD(v28) = v28 + 1;
  }
  while ( (__int64 *)v8 != v7 );
LABEL_15:
  v21 = 0;
  if ( (_BYTE)v9 )
  {
    a2 = v27;
    v21 = sub_B9C770(a1[1], v27, (__int64 *)(unsigned int)v28, 0, 1);
  }
  if ( v27 != (__int64 *)v29 )
    _libc_free(v27, a2);
  return v21;
}
