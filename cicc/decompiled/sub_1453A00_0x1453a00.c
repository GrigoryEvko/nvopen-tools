// Function: sub_1453A00
// Address: 0x1453a00
//
void __fastcall sub_1453A00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  const void *v5; // r8
  signed __int64 v6; // r12
  __int64 v7; // rbx
  char *v8; // rdi
  const void *v9; // [rsp+8h] [rbp-C8h]
  char *v10; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-B8h]
  _BYTE dest[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(const void **)(a2 + 8);
  v10 = dest;
  v11 = 0x2000000000LL;
  v6 = 4 * v4;
  v7 = v6 >> 2;
  if ( (unsigned __int64)v6 > 0x80 )
  {
    v9 = v5;
    sub_16CD150(&v10, dest, v6 >> 2, 4);
    v5 = v9;
    v8 = &v10[4 * (unsigned int)v11];
  }
  else
  {
    if ( !v6 )
      goto LABEL_3;
    v8 = dest;
  }
  memcpy(v8, v5, v6);
  LODWORD(v6) = v11;
LABEL_3:
  LODWORD(v11) = v6 + v7;
  sub_1453400(a3, &v10);
  if ( v10 != dest )
    _libc_free((unsigned __int64)v10);
}
