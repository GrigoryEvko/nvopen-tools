// Function: sub_105CEA0
// Address: 0x105cea0
//
__int64 __fastcall sub_105CEA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 result; // rax
  _BYTE *v14; // [rsp+0h] [rbp-190h] BYREF
  __int64 v15; // [rsp+8h] [rbp-188h]
  _BYTE v16[48]; // [rsp+10h] [rbp-180h] BYREF
  __int64 v17; // [rsp+40h] [rbp-150h] BYREF
  char *v18; // [rsp+48h] [rbp-148h]
  __int64 v19; // [rsp+50h] [rbp-140h]
  int v20; // [rsp+58h] [rbp-138h]
  char v21; // [rsp+5Ch] [rbp-134h]
  char v22; // [rsp+60h] [rbp-130h] BYREF

  v6 = *a2;
  v18 = &v22;
  v17 = 0;
  v15 = 0x600000000LL;
  v19 = 32;
  v20 = 0;
  v21 = 1;
  v14 = v16;
  sub_C8D5F0((__int64)&v14, v16, 0x18u, 8u, a5, a6);
  v9 = *(_QWORD *)(v6 + 80);
  v10 = v9 - 24;
  if ( !v9 )
    v10 = 0;
  v11 = (unsigned int)v15;
  v12 = (unsigned int)v15 + 1LL;
  if ( v12 > HIDWORD(v15) )
  {
    sub_C8D5F0((__int64)&v14, v16, v12, 8u, v7, v8);
    v11 = (unsigned int)v15;
  }
  *(_QWORD *)&v14[8 * v11] = v10;
  LODWORD(v15) = v15 + 1;
  result = sub_105C640(a1, (__int64)&v14, (__int64)a2, 0, (__int64)&v17);
  if ( v14 != v16 )
    result = _libc_free(v14, &v14);
  if ( !v21 )
    return _libc_free(v18, &v14);
  return result;
}
