// Function: sub_9AF960
// Address: 0x9af960
//
__int64 __fastcall sub_9AF960(__int64 a1, __int64 a2, __m128i *a3)
{
  int v4; // r12d
  int v5; // ebx
  unsigned int v6; // r12d
  unsigned int v7; // ebx
  __int64 result; // rax
  __int64 v9; // rdx
  unsigned int v10; // esi
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h]
  unsigned int v15; // [rsp+18h] [rbp-58h]
  __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-48h]
  __int64 v18; // [rsp+30h] [rbp-40h]
  unsigned int v19; // [rsp+38h] [rbp-38h]

  v4 = sub_BCB060(*(_QWORD *)(a1 + 8));
  v5 = sub_9AF7E0(a1, 0, a3);
  v6 = v4 + 1;
  v7 = sub_9AF7E0(a2, 0, a3) + v5;
  result = 3;
  if ( v6 >= v7 )
  {
    result = 2;
    if ( v6 == v7 )
    {
      sub_9AC330((__int64)&v12, a1, 0, a3);
      sub_9AC330((__int64)&v16, a2, 0, a3);
      if ( v13 > 0x40 )
        v9 = *(_QWORD *)(v12 + 8LL * ((v13 - 1) >> 6));
      else
        v9 = v12;
      if ( (v9 & (1LL << ((unsigned __int8)v13 - 1))) != 0 )
        goto LABEL_24;
      v10 = v17;
      v11 = v16;
      if ( v17 > 0x40 )
        v11 = *(_QWORD *)(v16 + 8LL * ((v17 - 1) >> 6));
      if ( (v11 & (1LL << ((unsigned __int8)v17 - 1))) != 0 )
      {
LABEL_24:
        if ( v19 > 0x40 && v18 )
          j_j___libc_free_0_0(v18);
        if ( v17 > 0x40 && v16 )
          j_j___libc_free_0_0(v16);
        if ( v15 > 0x40 && v14 )
          j_j___libc_free_0_0(v14);
        if ( v13 > 0x40 && v12 )
          j_j___libc_free_0_0(v12);
        return 3;
      }
      else
      {
        if ( v19 > 0x40 && v18 )
        {
          j_j___libc_free_0_0(v18);
          v10 = v17;
        }
        if ( v10 > 0x40 && v16 )
          j_j___libc_free_0_0(v16);
        if ( v15 > 0x40 && v14 )
          j_j___libc_free_0_0(v14);
        if ( v13 > 0x40 )
        {
          if ( v12 )
            j_j___libc_free_0_0(v12);
        }
        return 2;
      }
    }
  }
  return result;
}
