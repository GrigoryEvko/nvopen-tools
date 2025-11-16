// Function: sub_AD66B0
// Address: 0xad66b0
//
__int64 __fastcall sub_AD66B0(unsigned int a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r14d
  __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  __int64 result; // rax
  unsigned int v7; // eax
  unsigned int v8; // ebx
  __int64 v9; // r14
  __int64 v10; // [rsp+8h] [rbp-38h]
  unsigned __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  if ( a1 == 365 )
    return sub_AD6530(a2, a2);
  if ( a1 > 0x16D )
  {
    if ( a1 == 366 )
      return sub_AD62B0(a2);
    return 0;
  }
  if ( a1 == 329 )
  {
    v7 = *(_DWORD *)(a2 + 8) >> 8;
    v8 = v7 - 1;
    v12 = v7;
    v9 = 1LL << ((unsigned __int8)v7 - 1);
    if ( v7 > 0x40 )
    {
      sub_C43690(&v11, 0, 0);
      if ( v12 > 0x40 )
      {
        *(_QWORD *)(v11 + 8LL * (v8 >> 6)) |= v9;
        goto LABEL_18;
      }
    }
    else
    {
      v11 = 0;
    }
    v11 |= v9;
    goto LABEL_18;
  }
  if ( a1 != 330 )
    return 0;
  v2 = *(_DWORD *)(a2 + 8) >> 8;
  v3 = v2 - 1;
  v12 = v2;
  v4 = ~(1LL << ((unsigned __int8)v2 - 1));
  if ( v2 <= 0x40 )
  {
    v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v5 = 0;
    v11 = v5;
    goto LABEL_9;
  }
  sub_C43690(&v11, -1, 1);
  if ( v12 <= 0x40 )
  {
LABEL_9:
    v11 &= v4;
    goto LABEL_18;
  }
  *(_QWORD *)(v11 + 8LL * (v3 >> 6)) &= v4;
LABEL_18:
  result = sub_AD6220(a2, (__int64)&v11);
  if ( v12 > 0x40 )
  {
    if ( v11 )
    {
      v10 = result;
      j_j___libc_free_0_0(v11);
      return v10;
    }
  }
  return result;
}
