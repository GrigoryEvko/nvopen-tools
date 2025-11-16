// Function: sub_2787ED0
// Address: 0x2787ed0
//
void __fastcall sub_2787ED0(__int64 a1)
{
  __int64 v2; // r15
  __int64 v3; // r13
  bool v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  unsigned __int8 *v7; // r15
  __int64 v8; // rdx
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v13; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-B8h]
  const void *v15; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v17; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-98h]
  char v19; // [rsp+50h] [rbp-90h]
  __int64 v20[2]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v21; // [rsp+70h] [rbp-70h]
  _QWORD *v22; // [rsp+78h] [rbp-68h]
  __int64 v23; // [rsp+80h] [rbp-60h]
  _QWORD *v24; // [rsp+88h] [rbp-58h]
  _QWORD *v25; // [rsp+90h] [rbp-50h]
  __int64 v26; // [rsp+98h] [rbp-48h]
  __int64 v27; // [rsp+A0h] [rbp-40h]
  __int64 *v28; // [rsp+A8h] [rbp-38h]

  v20[0] = 0;
  v20[1] = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  sub_2785050(v20, 0);
  v2 = *(_QWORD *)(a1 + 32);
  v3 = v2 + 40LL * *(unsigned int *)(a1 + 40);
  if ( v2 != v3 )
  {
    while ( 1 )
    {
      sub_2784F00((__int64)&v15);
      if ( *(_DWORD *)(v2 + 16) <= 0x40u )
        break;
      v4 = sub_C43C50(v2 + 8, &v15);
      if ( v4 )
      {
        if ( *(_DWORD *)(v2 + 32) <= 0x40u )
          goto LABEL_17;
LABEL_5:
        v4 = sub_C43C50(v2 + 24, (const void **)&v17);
      }
LABEL_6:
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0((unsigned __int64)v15);
      if ( !v4 )
        goto LABEL_13;
      v5 = v25;
      if ( v25 == (_QWORD *)(v27 - 8) )
      {
        sub_2785520((unsigned __int64 *)v20, (_QWORD *)v2);
LABEL_13:
        v2 += 40;
        if ( v3 == v2 )
          goto LABEL_22;
      }
      else
      {
        if ( v25 )
        {
          *v25 = *(_QWORD *)v2;
          v5 = v25;
        }
        v2 += 40;
        v25 = v5 + 1;
        if ( v3 == v2 )
          goto LABEL_22;
      }
    }
    v4 = 0;
    if ( *(const void **)(v2 + 8) != v15 )
      goto LABEL_6;
    if ( *(_DWORD *)(v2 + 32) <= 0x40u )
    {
LABEL_17:
      v4 = *(_QWORD *)(v2 + 24) == v17;
      goto LABEL_6;
    }
    goto LABEL_5;
  }
LABEL_22:
  v6 = (unsigned __int64)v25;
  while ( v25 != v21 )
  {
    while ( 1 )
    {
      if ( v6 == v26 )
      {
        v7 = *(unsigned __int8 **)(*(v28 - 1) + 504);
        j_j___libc_free_0(v6);
        v8 = *--v28 + 512;
        v26 = *v28;
        v27 = v8;
        v25 = (_QWORD *)(v26 + 504);
      }
      else
      {
        v7 = *(unsigned __int8 **)(v6 - 8);
        v25 = (_QWORD *)(v6 - 8);
      }
      sub_2785600((__int64)&v15, a1, v7);
      if ( !v19 )
        break;
      v12 = v16;
      if ( v16 > 0x40 )
        sub_C43780((__int64)&v11, &v15);
      else
        v11 = (unsigned __int64)v15;
      v14 = v18;
      if ( v18 > 0x40 )
        sub_C43780((__int64)&v13, (const void **)&v17);
      else
        v13 = v17;
      sub_2787A60(a1, (__int64)v7, (__int64)&v11);
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      if ( v12 > 0x40 && v11 )
        j_j___libc_free_0_0(v11);
LABEL_34:
      if ( !v19 )
        goto LABEL_35;
LABEL_41:
      v19 = 0;
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 <= 0x40 || !v15 )
        goto LABEL_35;
      j_j___libc_free_0_0((unsigned __int64)v15);
      v6 = (unsigned __int64)v25;
      if ( v25 == v21 )
        goto LABEL_47;
    }
    if ( v21 == v22 )
    {
      v9 = v24;
      if ( ((v28 - v24 - 1) << 6) + (((__int64)v25 - v26) >> 3) + ((v23 - (__int64)v21) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( v24 == (_QWORD *)v20[0] )
      {
        sub_27853A0((unsigned __int64 *)v20, 1u, 1);
        v9 = v24;
      }
      *(v9 - 1) = sub_22077B0(0x200u);
      v10 = *--v24 + 512LL;
      v22 = (_QWORD *)*v24;
      v23 = v10;
      v21 = v22 + 63;
      v22[63] = v7;
      goto LABEL_34;
    }
    *--v21 = v7;
    if ( v19 )
      goto LABEL_41;
LABEL_35:
    v6 = (unsigned __int64)v25;
  }
LABEL_47:
  sub_2784FD0((unsigned __int64 *)v20);
}
