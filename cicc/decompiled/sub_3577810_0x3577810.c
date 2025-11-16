// Function: sub_3577810
// Address: 0x3577810
//
void __fastcall sub_3577810(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // rdx
  _BYTE *v12; // [rsp+0h] [rbp-190h] BYREF
  __int64 v13; // [rsp+8h] [rbp-188h]
  _BYTE v14[48]; // [rsp+10h] [rbp-180h] BYREF
  __int64 v15; // [rsp+40h] [rbp-150h] BYREF
  char *v16; // [rsp+48h] [rbp-148h]
  __int64 v17; // [rsp+50h] [rbp-140h]
  int v18; // [rsp+58h] [rbp-138h]
  char v19; // [rsp+5Ch] [rbp-134h]
  char v20; // [rsp+60h] [rbp-130h] BYREF

  v16 = &v20;
  v6 = *a2;
  v13 = 0x600000000LL;
  v15 = 0;
  v17 = 32;
  v18 = 0;
  v19 = 1;
  v12 = v14;
  sub_C8D5F0((__int64)&v12, v14, 0x18u, 8u, a5, a6);
  v9 = (unsigned int)v13;
  v10 = *(_QWORD *)(v6 + 328);
  v11 = (unsigned int)v13 + 1LL;
  if ( v11 > HIDWORD(v13) )
  {
    sub_C8D5F0((__int64)&v12, v14, v11, 8u, v7, v8);
    v9 = (unsigned int)v13;
  }
  *(_QWORD *)&v12[8 * v9] = v10;
  LODWORD(v13) = v13 + 1;
  sub_3576F90(a1, (__int64)&v12, (__int64)a2, 0, (__int64)&v15);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  if ( !v19 )
    _libc_free((unsigned __int64)v16);
}
