// Function: sub_1949C30
// Address: 0x1949c30
//
__int64 __fastcall sub_1949C30(
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
  unsigned int v13; // r15d
  __int64 v14; // rax
  __int64 *v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // [rsp+0h] [rbp-70h]
  bool v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  unsigned __int8 v27; // [rsp+18h] [rbp-58h]
  unsigned __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+28h] [rbp-48h]
  _QWORD v30[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( ((a4 - 36) & 0xFFFFFFFB) != 0 && (a4 & 0xFFFFFFFB) != 0x22 || !sub_146D950(a9, a2, a6) )
    return 0;
  v24 = sub_15FF7F0(a4);
  v13 = !v24 ? 34 : 38;
  if ( a5 == 1 )
    return sub_148B410(a9, a6, v13, a1, a2);
  v14 = sub_1456040(a3);
  v30[1] = sub_145CF80(a9, v14, 1, 0);
  v28 = (unsigned __int64)v30;
  v30[0] = a3;
  v29 = 0x200000002LL;
  v15 = sub_147DD40(a9, (__int64 *)&v28, 0, 0, a7, a8);
  if ( (_QWORD *)v28 != v30 )
    _libc_free(v28);
  v16 = *(_DWORD *)(sub_1456040(a2) + 8) >> 8;
  LODWORD(v29) = v16;
  if ( !v24 )
  {
    if ( v16 > 0x40 )
      sub_16A4EF0((__int64)&v28, 0, 0);
    else
      v28 = 0;
    goto LABEL_12;
  }
  v22 = 1LL << ((unsigned __int8)v16 - 1);
  if ( v16 > 0x40 )
  {
    v23 = v16 - 1;
    v25 = 1LL << ((unsigned __int8)v16 - 1);
    sub_16A4EF0((__int64)&v28, 0, 0);
    v22 = v25;
    if ( (unsigned int)v29 > 0x40 )
    {
      *(_QWORD *)(v28 + 8LL * (v23 >> 6)) |= v25;
      goto LABEL_12;
    }
  }
  else
  {
    v28 = 0;
  }
  v28 |= v22;
LABEL_12:
  v17 = sub_145CF40(a9, (__int64)&v28);
  v18 = sub_14806B0(a9, v17, (__int64)v15, 0, 0);
  v19 = sub_1456040(a2);
  v20 = sub_145CF80(a9, v19, 1, 0);
  v21 = sub_14806B0(a9, a2, v20, 0, 0);
  result = sub_148B410(a9, a6, v13, a1, v21);
  if ( (_BYTE)result )
    result = sub_148B410(a9, a6, v13, a2, v18);
  if ( (unsigned int)v29 > 0x40 )
  {
    if ( v28 )
    {
      v27 = result;
      j_j___libc_free_0_0(v28);
      return v27;
    }
  }
  return result;
}
