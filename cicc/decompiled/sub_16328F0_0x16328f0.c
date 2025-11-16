// Function: sub_16328F0
// Address: 0x16328f0
//
__int64 __fastcall sub_16328F0(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int64 v4; // r12
  _BYTE *v5; // r13
  const void *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  _BYTE *v10; // [rsp+0h] [rbp-100h] BYREF
  __int64 v11; // [rsp+8h] [rbp-F8h]
  _BYTE v12[240]; // [rsp+10h] [rbp-F0h] BYREF

  v11 = 0x800000000LL;
  v10 = v12;
  sub_16327D0(a1, (__int64)&v10);
  v4 = (unsigned __int64)v10;
  v5 = &v10[24 * (unsigned int)v11];
  if ( v5 == v10 )
  {
    v8 = 0;
  }
  else
  {
    while ( 1 )
    {
      v6 = (const void *)sub_161E970(*(_QWORD *)(v4 + 8));
      if ( a3 == v7 && (!a3 || !memcmp(a2, v6, a3)) )
        break;
      v4 += 24LL;
      if ( (_BYTE *)v4 == v5 )
      {
        v4 = (unsigned __int64)v10;
        v8 = 0;
        goto LABEL_8;
      }
    }
    v8 = *(_QWORD *)(v4 + 16);
    v4 = (unsigned __int64)v10;
  }
LABEL_8:
  if ( (_BYTE *)v4 != v12 )
    _libc_free(v4);
  return v8;
}
