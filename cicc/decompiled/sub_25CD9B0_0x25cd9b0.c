// Function: sub_25CD9B0
// Address: 0x25cd9b0
//
unsigned __int64 *__fastcall sub_25CD9B0(unsigned __int64 *a1, __int64 a2, __int64 a3, __m128i a4)
{
  const char **v6; // r11
  __int64 v7; // rdx
  _BYTE *v8; // rbx
  __int64 v9; // r8
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rdi
  _QWORD *v13; // rax
  _QWORD v14[2]; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD *v15; // [rsp+20h] [rbp-190h]
  __int64 v16; // [rsp+28h] [rbp-188h]
  _QWORD v17[3]; // [rsp+30h] [rbp-180h] BYREF
  int v18; // [rsp+48h] [rbp-168h]
  _QWORD *v19; // [rsp+50h] [rbp-160h]
  __int64 v20; // [rsp+58h] [rbp-158h]
  _BYTE v21[16]; // [rsp+60h] [rbp-150h] BYREF
  _QWORD *v22; // [rsp+70h] [rbp-140h]
  __int64 v23; // [rsp+78h] [rbp-138h]
  _BYTE v24[16]; // [rsp+80h] [rbp-130h] BYREF
  unsigned __int64 v25; // [rsp+90h] [rbp-120h]
  __int64 v26; // [rsp+98h] [rbp-118h]
  __int64 v27; // [rsp+A0h] [rbp-110h]
  _BYTE *v28; // [rsp+A8h] [rbp-108h]
  __int64 v29; // [rsp+B0h] [rbp-100h]
  _BYTE v30[248]; // [rsp+B8h] [rbp-F8h] BYREF

  v6 = *(const char ***)a2;
  v7 = *(_QWORD *)(a2 + 8);
  v15 = v17;
  v19 = v21;
  v14[0] = 0;
  v14[1] = 0;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  v17[2] = 0;
  v18 = 0;
  v20 = 0;
  v21[0] = 0;
  v22 = v24;
  v23 = 0;
  v24[0] = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = v30;
  v29 = 0x400000000LL;
  sub_E477F0(a1, v6, v7, (__int64)v14, a3, 1u, a4);
  if ( !*a1 )
  {
    v13 = sub_CB72A0();
    sub_C8EE80((__int64)v14, "function-import", v13, 1u, 1, 1);
    sub_C64ED0("Abort", 1u);
  }
  v8 = v28;
  v9 = 48LL * (unsigned int)v29;
  v10 = (unsigned __int64)&v28[v9];
  if ( v28 != &v28[v9] )
  {
    do
    {
      v10 -= 48LL;
      v11 = *(_QWORD *)(v10 + 16);
      if ( v11 != v10 + 32 )
        j_j___libc_free_0(v11);
    }
    while ( v8 != (_BYTE *)v10 );
    v10 = (unsigned __int64)v28;
  }
  if ( (_BYTE *)v10 != v30 )
    _libc_free(v10);
  if ( v25 )
    j_j___libc_free_0(v25);
  if ( v22 != (_QWORD *)v24 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( v19 != (_QWORD *)v21 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( v15 != v17 )
    j_j___libc_free_0((unsigned __int64)v15);
  return a1;
}
