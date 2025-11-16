// Function: sub_39D3680
// Address: 0x39d3680
//
__int64 __fastcall sub_39D3680(__int64 a1, _QWORD *a2)
{
  int v2; // r9d
  _BYTE *v3; // r14
  __int64 v4; // rcx
  const void *v5; // rdi
  unsigned int v6; // r12d
  signed __int64 v7; // rdx
  __int64 v9; // rbx
  _QWORD *v10; // rdi
  __int64 v11; // r8
  _QWORD *v12; // rax
  char v13; // [rsp+Fh] [rbp-81h] BYREF
  void *s2; // [rsp+10h] [rbp-80h] BYREF
  __int64 v15; // [rsp+18h] [rbp-78h]
  _BYTE v16[112]; // [rsp+20h] [rbp-70h] BYREF

  s2 = v16;
  v15 = 0x800000000LL;
  sub_39D3400((__int64)a2, (__int64)&s2, &v13);
  if ( !v13 )
  {
    v3 = s2;
    v4 = (unsigned int)v15;
    goto LABEL_3;
  }
  v4 = (unsigned int)v15;
  v9 = a2[1];
  v3 = s2;
  if ( v9 == a2[7] + 320LL )
    goto LABEL_3;
  v10 = (char *)s2 + 8 * (unsigned int)v15;
  v11 = (8LL * (unsigned int)v15) >> 3;
  if ( (8LL * (unsigned int)v15) >> 5 )
  {
    v12 = s2;
    while ( v9 != *v12 )
    {
      if ( v9 == v12[1] )
      {
        ++v12;
        break;
      }
      if ( v9 == v12[2] )
      {
        v12 += 2;
        break;
      }
      if ( v9 == v12[3] )
      {
        v12 += 3;
        break;
      }
      v12 += 4;
      if ( (char *)s2 + 32 * ((8LL * (unsigned int)v15) >> 5) == (char *)v12 )
      {
        v11 = v10 - v12;
        goto LABEL_25;
      }
    }
LABEL_17:
    if ( v10 != v12 )
      goto LABEL_3;
    goto LABEL_18;
  }
  v12 = s2;
LABEL_25:
  if ( v11 != 2 )
  {
    if ( v11 != 3 )
    {
      if ( v11 != 1 )
        goto LABEL_18;
      goto LABEL_28;
    }
    if ( v9 == *v12 )
      goto LABEL_17;
    ++v12;
  }
  if ( v9 == *v12 )
    goto LABEL_17;
  ++v12;
LABEL_28:
  if ( v9 == *v12 )
    goto LABEL_17;
LABEL_18:
  if ( HIDWORD(v15) <= (unsigned int)v15 )
  {
    sub_16CD150((__int64)&s2, v16, 0, 8, v11, v2);
    v10 = (char *)s2 + 8 * (unsigned int)v15;
  }
  *v10 = v9;
  v3 = s2;
  v4 = (unsigned int)(v15 + 1);
  LODWORD(v15) = v15 + 1;
LABEL_3:
  v5 = (const void *)a2[11];
  v6 = 0;
  v7 = a2[12] - (_QWORD)v5;
  if ( (unsigned int)(v7 >> 3) == v4 )
  {
    v6 = 1;
    if ( v7 )
      LOBYTE(v6) = memcmp(v5, v3, v7) == 0;
  }
  if ( v3 != v16 )
    _libc_free((unsigned __int64)v3);
  return v6;
}
