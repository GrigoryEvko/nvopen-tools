// Function: sub_1949930
// Address: 0x1949930
//
__int64 __fastcall sub_1949930(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        int a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9)
{
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 v22; // [rsp+0h] [rbp-80h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  unsigned int v24; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned int v28; // [rsp+18h] [rbp-68h]
  unsigned __int8 v29; // [rsp+18h] [rbp-68h]
  unsigned __int64 v30; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-58h]
  __int64 v32[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v33[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( ((a4 - 36) & 0xFFFFFFFB) != 0 && (a4 & 0xFFFFFFFB) != 0x22 || !sub_146D950(a9, a2, a6) )
    return 0;
  if ( !sub_15FF7F0(a4) )
  {
    if ( a5 != 1 )
    {
      v12 = sub_1456040(a3);
      v13 = sub_145CF80(a9, v12, 1, 0);
      v26 = sub_14806B0(a9, a3, v13, 0, 0);
      v14 = *(_DWORD *)(sub_1456040(a2) + 8) >> 8;
      v31 = v14;
      if ( v14 > 0x40 )
      {
        sub_16A4EF0((__int64)&v30, -1, 1);
        v28 = 36;
      }
      else
      {
        v28 = 36;
        v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
      }
      goto LABEL_10;
    }
    v21 = 36;
    return sub_148B410(a9, a6, v21, a1, a2);
  }
  if ( a5 == 1 )
  {
    v21 = 40;
    return sub_148B410(a9, a6, v21, a1, a2);
  }
  v17 = sub_1456040(a3);
  v18 = sub_145CF80(a9, v17, 1, 0);
  v26 = sub_14806B0(a9, a3, v18, 0, 0);
  v19 = *(_DWORD *)(sub_1456040(a2) + 8) >> 8;
  v31 = v19;
  v20 = ~(1LL << ((unsigned __int8)v19 - 1));
  if ( v19 > 0x40 )
  {
    v22 = ~(1LL << ((unsigned __int8)v19 - 1));
    v24 = v19 - 1;
    sub_16A4EF0((__int64)&v30, -1, 1);
    v20 = v22;
    if ( v31 > 0x40 )
    {
      v28 = 40;
      *(_QWORD *)(v30 + 8LL * (v24 >> 6)) &= v22;
      goto LABEL_10;
    }
  }
  else
  {
    v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
  }
  v30 &= v20;
  v28 = 40;
LABEL_10:
  v15 = sub_145CF40(a9, (__int64)&v30);
  v23 = sub_14806B0(a9, v15, v26, 0, 0);
  v32[0] = (__int64)v33;
  v33[1] = a3;
  v33[0] = a2;
  v32[1] = 0x200000002LL;
  v16 = sub_147DD40(a9, v32, 0, 0, a7, a8);
  if ( (_QWORD *)v32[0] != v33 )
    _libc_free(v32[0]);
  result = sub_148B410(a9, a6, v28, a1, (__int64)v16);
  if ( (_BYTE)result )
    result = sub_148B410(a9, a6, v28, a2, v23);
  if ( v31 > 0x40 )
  {
    if ( v30 )
    {
      v29 = result;
      j_j___libc_free_0_0(v30);
      return v29;
    }
  }
  return result;
}
