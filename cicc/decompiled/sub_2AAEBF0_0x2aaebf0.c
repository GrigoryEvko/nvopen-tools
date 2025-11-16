// Function: sub_2AAEBF0
// Address: 0x2aaebf0
//
__int64 __fastcall sub_2AAEBF0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // eax
  unsigned int v3; // ebx
  __int64 v4; // r12
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-38h]
  unsigned __int64 v7; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-28h]

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  v2 = *(_DWORD *)(v1 + 8) >> 8;
  v3 = v2 - 1;
  v8 = v2;
  v4 = 1LL << ((unsigned __int8)v2 - 1);
  if ( v2 <= 0x40 )
  {
    v7 = 0;
LABEL_3:
    v7 |= v4;
    goto LABEL_4;
  }
  sub_C43690((__int64)&v7, 0, 0);
  if ( v8 <= 0x40 )
    goto LABEL_3;
  *(_QWORD *)(v7 + 8LL * (v3 >> 6)) |= v4;
LABEL_4:
  result = sub_AD8D80(v1, (__int64)&v7);
  if ( v8 > 0x40 )
  {
    if ( v7 )
    {
      v6 = result;
      j_j___libc_free_0_0(v7);
      return v6;
    }
  }
  return result;
}
