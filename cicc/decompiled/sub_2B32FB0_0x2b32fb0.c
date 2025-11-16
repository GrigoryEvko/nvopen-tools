// Function: sub_2B32FB0
// Address: 0x2b32fb0
//
void __fastcall sub_2B32FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r14
  const void *v6; // r9
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  _BYTE *v11; // rdi
  const void *v12; // [rsp+0h] [rbp-80h]
  _BYTE *v13; // [rsp+10h] [rbp-70h] BYREF
  __int64 v14; // [rsp+18h] [rbp-68h]
  _BYTE dest[96]; // [rsp+20h] [rbp-60h] BYREF

  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(const void **)a1;
  v13 = dest;
  v14 = 0xC00000000LL;
  v7 = 4 * v5;
  if ( v5 > 0xC )
  {
    v12 = v6;
    sub_C8D5F0((__int64)&v13, dest, v5, 4u, (__int64)&v13, (__int64)v6);
    v6 = v12;
    v11 = &v13[4 * (unsigned int)v14];
  }
  else
  {
    if ( !v7 )
      goto LABEL_3;
    v11 = dest;
  }
  memcpy(v11, v6, 4 * v5);
  LODWORD(v7) = v14;
LABEL_3:
  LODWORD(v14) = v7 + v5;
  sub_2B310D0((__int64)&v13, a1, a3, a4, (__int64)&v13, (__int64)v6);
  v8 = 0;
  v9 = 4LL * (unsigned int)v14;
  if ( (_DWORD)v14 )
  {
    do
    {
      v10 = *(int *)(a2 + v8);
      if ( (_DWORD)v10 != -1 )
        *(_DWORD *)(*(_QWORD *)a1 + 4 * v10) = *(_DWORD *)&v13[v8];
      v8 += 4;
    }
    while ( v9 != v8 );
  }
  if ( v13 != dest )
    _libc_free((unsigned __int64)v13);
}
