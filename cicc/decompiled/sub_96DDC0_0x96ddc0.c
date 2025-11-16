// Function: sub_96DDC0
// Address: 0x96ddc0
//
__int64 __fastcall sub_96DDC0(unsigned int a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r14d
  __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  unsigned int v6; // ebx
  __int64 v7; // r14
  unsigned __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // [rsp+8h] [rbp-38h]
  unsigned __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v2 = sub_BCB060(a2);
  if ( a1 == 365 )
  {
    v12 = v2;
    if ( v2 > 0x40 )
    {
      sub_C43690(&v11, -1, 1);
    }
    else
    {
      v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
      if ( !v2 )
        v8 = 0;
      v11 = v8;
    }
    goto LABEL_20;
  }
  if ( a1 > 0x16D )
  {
    if ( a1 == 366 )
    {
      v12 = v2;
      if ( v2 > 0x40 )
        sub_C43690(&v11, 0, 0);
      else
        v11 = 0;
      goto LABEL_20;
    }
LABEL_30:
    BUG();
  }
  if ( a1 != 329 )
  {
    if ( a1 == 330 )
    {
      v6 = v2 - 1;
      v12 = v2;
      v7 = 1LL << ((unsigned __int8)v2 - 1);
      if ( v2 > 0x40 )
      {
        sub_C43690(&v11, 0, 0);
        if ( v12 > 0x40 )
        {
          *(_QWORD *)(v11 + 8LL * (v6 >> 6)) |= v7;
          goto LABEL_20;
        }
      }
      else
      {
        v11 = 0;
      }
      v11 |= v7;
      goto LABEL_20;
    }
    goto LABEL_30;
  }
  v3 = v2 - 1;
  v12 = v2;
  v4 = ~(1LL << ((unsigned __int8)v2 - 1));
  if ( v2 <= 0x40 )
  {
    v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v5 = 0;
    v11 = v5;
    goto LABEL_8;
  }
  sub_C43690(&v11, -1, 1);
  if ( v12 <= 0x40 )
  {
LABEL_8:
    v11 &= v4;
    goto LABEL_20;
  }
  *(_QWORD *)(v11 + 8LL * (v3 >> 6)) &= v4;
LABEL_20:
  result = sub_AD6220(a2, &v11);
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
