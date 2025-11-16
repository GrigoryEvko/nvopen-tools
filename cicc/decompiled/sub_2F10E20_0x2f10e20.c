// Function: sub_2F10E20
// Address: 0x2f10e20
//
__int64 __fastcall sub_2F10E20(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r9
  __int64 v5; // rdx
  _BYTE *v6; // r15
  unsigned int v7; // r14d
  size_t v8; // rdx
  __int64 v10; // rbx
  __int64 v11; // rcx
  char *v12; // rdi
  __int64 v13; // r8
  __int64 v14; // rcx
  char *v15; // rax
  char *v16; // rcx
  char v17; // [rsp+Fh] [rbp-81h] BYREF
  void *s2; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h]
  _BYTE v20[112]; // [rsp+20h] [rbp-70h] BYREF

  s2 = v20;
  v19 = 0x800000000LL;
  sub_2F10BD0(a2, (__int64)&s2, (__int64 *)&v17, a4);
  if ( !v17 )
  {
    v5 = (unsigned int)v19;
    v6 = s2;
    goto LABEL_3;
  }
  v10 = *(_QWORD *)(a2 + 8);
  v6 = s2;
  v5 = (unsigned int)v19;
  if ( v10 == *(_QWORD *)(a2 + 32) + 320LL )
    goto LABEL_3;
  v11 = 8LL * (unsigned int)v19;
  v12 = (char *)s2 + v11;
  v13 = v11 >> 3;
  v14 = v11 >> 5;
  if ( v14 )
  {
    v15 = (char *)s2;
    v16 = (char *)s2 + 32 * v14;
    while ( *(_QWORD *)v15 != v10 )
    {
      if ( *((_QWORD *)v15 + 1) == v10 )
      {
        v15 += 8;
        break;
      }
      if ( *((_QWORD *)v15 + 2) == v10 )
      {
        v15 += 16;
        break;
      }
      if ( *((_QWORD *)v15 + 3) == v10 )
      {
        v15 += 24;
        break;
      }
      v15 += 32;
      if ( v15 == v16 )
      {
        v13 = (v12 - v15) >> 3;
        goto LABEL_25;
      }
    }
LABEL_17:
    if ( v12 != v15 )
      goto LABEL_3;
    goto LABEL_18;
  }
  v15 = (char *)s2;
LABEL_25:
  if ( v13 != 2 )
  {
    if ( v13 != 3 )
    {
      if ( v13 != 1 )
        goto LABEL_18;
      goto LABEL_28;
    }
    if ( *(_QWORD *)v15 == v10 )
      goto LABEL_17;
    v15 += 8;
  }
  if ( *(_QWORD *)v15 == v10 )
    goto LABEL_17;
  v15 += 8;
LABEL_28:
  if ( *(_QWORD *)v15 == v10 )
    goto LABEL_17;
LABEL_18:
  if ( (unsigned __int64)(unsigned int)v19 + 1 > HIDWORD(v19) )
  {
    sub_C8D5F0((__int64)&s2, v20, (unsigned int)v19 + 1LL, 8u, v13, v4);
    v12 = (char *)s2 + 8 * (unsigned int)v19;
  }
  *(_QWORD *)v12 = v10;
  v6 = s2;
  v5 = (unsigned int)(v19 + 1);
  LODWORD(v19) = v19 + 1;
LABEL_3:
  v7 = 0;
  if ( *(_DWORD *)(a2 + 120) == (_DWORD)v5 )
  {
    v8 = 8 * v5;
    v7 = 1;
    if ( v8 )
      LOBYTE(v7) = memcmp(*(const void **)(a2 + 112), v6, v8) == 0;
  }
  if ( v6 != v20 )
    _libc_free((unsigned __int64)v6);
  return v7;
}
