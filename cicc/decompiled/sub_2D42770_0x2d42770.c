// Function: sub_2D42770
// Address: 0x2d42770
//
__int64 __fastcall sub_2D42770(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE *v10; // r13
  _BYTE *v11; // r12
  __int64 v12; // rsi
  _QWORD *v13; // rbx
  unsigned __int64 v14; // r13
  __int64 v15; // r15
  unsigned __int64 v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  char *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r13
  void *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-E0h] BYREF
  int v25; // [rsp+8h] [rbp-D8h] BYREF
  unsigned __int64 v26; // [rsp+10h] [rbp-D0h]
  int *v27; // [rsp+18h] [rbp-C8h]
  int *v28; // [rsp+20h] [rbp-C0h]
  __int64 v29; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v30; // [rsp+30h] [rbp-B0h]
  __int64 v31; // [rsp+38h] [rbp-A8h]
  __int64 v32; // [rsp+40h] [rbp-A0h]
  void *s; // [rsp+48h] [rbp-98h]
  __int64 v34; // [rsp+50h] [rbp-90h]
  _QWORD *v35; // [rsp+58h] [rbp-88h]
  __int64 v36; // [rsp+60h] [rbp-80h]
  int v37; // [rsp+68h] [rbp-78h]
  __int64 v38; // [rsp+70h] [rbp-70h]
  __int64 v39; // [rsp+78h] [rbp-68h] BYREF
  _BYTE *v40; // [rsp+80h] [rbp-60h]
  __int64 v41; // [rsp+88h] [rbp-58h]
  _BYTE v42[80]; // [rsp+90h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_AEA460(*(_QWORD *)(a2 + 40)) )
  {
    sub_2D281A0(*(_QWORD *)(a1 + 176));
    v25 = 0;
    v27 = &v25;
    v28 = &v25;
    v26 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    s = &v39;
    v34 = 1;
    v35 = 0;
    v36 = 0;
    v37 = 1065353216;
    v38 = 0;
    v39 = 0;
    v40 = v42;
    v41 = 0x100000000LL;
    v5 = sub_B2BEC0(a2);
    sub_2D3F710(a2, v5, &v24, a3, a4);
    sub_2D2C8E0(*(_QWORD *)(a1 + 176), (__int64)&v24, v6, v7, v8, v9);
    if ( (_BYTE)qword_5016748 )
    {
      v20 = (char *)sub_BD5D20(a2);
      if ( sub_BC63A0(v20, v21) )
      {
        v22 = *(__int64 **)(a1 + 176);
        v23 = sub_CB72A0();
        sub_2D27C30(v22, (__int64)v23, a2);
      }
    }
    v10 = v40;
    v11 = &v40[32 * (unsigned int)v41];
    if ( v40 != v11 )
    {
      do
      {
        v12 = *((_QWORD *)v11 - 2);
        v11 -= 32;
        if ( v12 )
          sub_B91220((__int64)(v11 + 16), v12);
      }
      while ( v10 != v11 );
      v11 = v40;
    }
    if ( v11 != v42 )
      _libc_free((unsigned __int64)v11);
    v13 = v35;
    while ( v13 )
    {
      v14 = (unsigned __int64)v13;
      v13 = (_QWORD *)*v13;
      v15 = *(_QWORD *)(v14 + 16);
      v16 = v15 + 32LL * *(unsigned int *)(v14 + 24);
      if ( v15 != v16 )
      {
        do
        {
          v17 = *(_QWORD *)(v16 - 16);
          v16 -= 32LL;
          if ( v17 )
            sub_B91220(v16 + 16, v17);
        }
        while ( v15 != v16 );
        v16 = *(_QWORD *)(v14 + 16);
      }
      if ( v16 != v14 + 32 )
        _libc_free(v16);
      j_j___libc_free_0(v14);
    }
    memset(s, 0, 8 * v34);
    v36 = 0;
    v35 = 0;
    if ( s != &v39 )
      j_j___libc_free_0((unsigned __int64)s);
    if ( v30 )
      j_j___libc_free_0(v30);
    v18 = v26;
    while ( v18 )
    {
      sub_2D24760(*(_QWORD *)(v18 + 24));
      v19 = v18;
      v18 = *(_QWORD *)(v18 + 16);
      j_j___libc_free_0(v19);
    }
  }
  return 0;
}
