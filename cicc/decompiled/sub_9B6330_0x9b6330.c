// Function: sub_9B6330
// Address: 0x9b6330
//
__int64 __fastcall sub_9B6330(__int64 a1, __m128i *a2, unsigned int a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v8; // rax
  unsigned int v9; // ebx
  unsigned __int8 v10; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v11; // [rsp+Fh] [rbp-41h]
  __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-38h]
  __int64 v14; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-28h]

  if ( *(_BYTE *)a1 == 17 )
  {
    v3 = *(_DWORD *)(a1 + 32);
    v4 = *(_QWORD *)(a1 + 24);
    v5 = 1LL << ((unsigned __int8)v3 - 1);
    if ( v3 > 0x40 )
    {
      if ( (*(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6)) & v5) != 0 )
        return 0;
      LODWORD(v5) = sub_C444A0(a1 + 24);
      LOBYTE(v5) = v3 == (_DWORD)v5;
    }
    else
    {
      if ( (v5 & v4) != 0 )
        return 0;
      LOBYTE(v5) = v4 == 0;
    }
    return (unsigned int)v5 ^ 1;
  }
  sub_9AC330((__int64)&v12, a1, a3, a2);
  v8 = 1LL << ((unsigned __int8)v13 - 1);
  if ( v13 > 0x40 )
  {
    v9 = v15;
    if ( (*(_QWORD *)(v12 + 8LL * ((v13 - 1) >> 6)) & v8) != 0 )
      goto LABEL_11;
LABEL_20:
    result = 0;
    goto LABEL_21;
  }
  v9 = v15;
  if ( (v12 & v8) == 0 )
    goto LABEL_20;
LABEL_11:
  if ( v9 <= 0x40 )
  {
    result = 1;
    if ( v14 )
      goto LABEL_16;
  }
  else if ( (unsigned int)sub_C444A0(&v14) != v9 )
  {
    result = 1;
    goto LABEL_14;
  }
  result = sub_9B6260(a1, a2, a3);
  v9 = v15;
LABEL_21:
  if ( v9 > 0x40 )
  {
LABEL_14:
    if ( v14 )
    {
      v10 = result;
      j_j___libc_free_0_0(v14);
      result = v10;
    }
  }
LABEL_16:
  if ( v13 > 0x40 )
  {
    if ( v12 )
    {
      v11 = result;
      j_j___libc_free_0_0(v12);
      return v11;
    }
  }
  return result;
}
