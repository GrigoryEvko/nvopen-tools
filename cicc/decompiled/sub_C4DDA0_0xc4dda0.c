// Function: sub_C4DDA0
// Address: 0xc4dda0
//
__int64 __fastcall sub_C4DDA0(__int64 a1, const void **a2)
{
  unsigned int v2; // ebx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  int v6; // edx
  unsigned __int64 v7; // r12
  int v8; // eax
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // [rsp+8h] [rbp-48h]
  unsigned __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-38h]
  unsigned __int64 v14; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-28h]

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 <= 0x40 )
  {
    v4 = *(_QWORD *)a1;
    v5 = (unsigned __int64)*a2;
    v6 = 63;
    if ( *(const void **)a1 == *a2 )
      goto LABEL_3;
    goto LABEL_6;
  }
  if ( sub_C43C50(a1, a2) )
  {
LABEL_3:
    BYTE4(v11) = 0;
    return v11;
  }
  v13 = v2;
  sub_C43780((__int64)&v12, (const void **)a1);
  if ( v13 <= 0x40 )
  {
    v4 = v12;
    v5 = (unsigned __int64)*a2;
    v6 = v2 - v13 + 63;
LABEL_6:
    v7 = v5 ^ v4;
    goto LABEL_7;
  }
  sub_C43C10(&v12, (__int64 *)a2);
  v10 = v13;
  v7 = v12;
  v13 = 0;
  v15 = v10;
  v14 = v12;
  if ( v10 > 0x40 )
  {
    BYTE4(v11) = 1;
    LODWORD(v11) = v2 - 1 - sub_C444A0((__int64)&v14);
    if ( v7 )
    {
      j_j___libc_free_0_0(v7);
      if ( v13 > 0x40 )
      {
        if ( v12 )
          j_j___libc_free_0_0(v12);
      }
    }
    return v11;
  }
  v6 = v2 - v10 + 63;
LABEL_7:
  v8 = 64;
  if ( v7 )
  {
    _BitScanReverse64(&v9, v7);
    v8 = v9 ^ 0x3F;
  }
  BYTE4(v11) = 1;
  LODWORD(v11) = v6 - v8;
  return v11;
}
