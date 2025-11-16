// Function: sub_D91960
// Address: 0xd91960
//
void __fastcall sub_D91960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  const void *v8; // r8
  signed __int64 v9; // r12
  __int64 v10; // rbx
  char *v11; // rdi
  const void *v12; // [rsp+8h] [rbp-C8h]
  char *v13; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+18h] [rbp-B8h]
  _BYTE dest[176]; // [rsp+20h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(const void **)(a2 + 8);
  v13 = dest;
  v14 = 0x2000000000LL;
  v9 = 4 * v7;
  v10 = v9 >> 2;
  if ( (unsigned __int64)v9 > 0x80 )
  {
    v12 = v8;
    sub_C8D5F0((__int64)&v13, dest, v9 >> 2, 4u, (__int64)v8, a6);
    v8 = v12;
    v11 = &v13[4 * (unsigned int)v14];
  }
  else
  {
    if ( !v9 )
      goto LABEL_3;
    v11 = dest;
  }
  memcpy(v11, v8, v9);
  LODWORD(v9) = v14;
LABEL_3:
  LODWORD(v14) = v9 + v10;
  sub_D916A0(a3, &v13, a3, a4, (__int64)v8, a6);
  if ( v13 != dest )
    _libc_free(v13, &v13);
}
