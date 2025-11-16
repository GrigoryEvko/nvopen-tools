// Function: sub_14876A0
// Address: 0x14876a0
//
__int64 __fastcall sub_14876A0(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v8; // rdx
  char v9; // cl
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 *v13; // r15
  __int64 v14; // r8
  __int64 v15; // rax
  int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // [rsp+8h] [rbp-68h]
  _BYTE *v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  _BYTE v23[80]; // [rsp+20h] [rbp-50h] BYREF

  v5 = sub_16348C0(a2);
  v6 = *(unsigned __int8 *)(v5 + 8);
  if ( (unsigned __int8)v6 > 0xFu || (v8 = 35454, !_bittest64(&v8, v6)) )
  {
    if ( (unsigned int)(v6 - 13) > 1 && (_DWORD)v6 != 16 || !(unsigned __int8)sub_16435F0(v5, 0) )
      return sub_145DC80((__int64)a1, a2);
  }
  v9 = *(_BYTE *)(a2 + 23);
  v22 = 0x400000000LL;
  v10 = *(_DWORD *)(a2 + 20);
  v21 = v23;
  v11 = v10 & 0xFFFFFFF;
  if ( (v9 & 0x40) != 0 )
    v12 = *(_QWORD *)(a2 - 8);
  else
    v12 = a2 - 24 * v11;
  v13 = (__int64 *)(v12 + 24);
  while ( 1 )
  {
    v17 = (__int64 *)a2;
    v18 = 24 * v11;
    if ( (v9 & 0x40) != 0 )
      v17 = (__int64 *)(*(_QWORD *)(a2 - 8) + v18);
    if ( v13 == v17 )
      break;
    v14 = sub_146F1B0((__int64)a1, *v13);
    v15 = (unsigned int)v22;
    if ( (unsigned int)v22 >= HIDWORD(v22) )
    {
      v20 = v14;
      sub_16CD150(&v21, v23, 0, 8);
      v15 = (unsigned int)v22;
      v14 = v20;
    }
    v13 += 3;
    *(_QWORD *)&v21[8 * v15] = v14;
    v16 = *(_DWORD *)(a2 + 20);
    LODWORD(v22) = v22 + 1;
    v9 = *(_BYTE *)(a2 + 23);
    v11 = v16 & 0xFFFFFFF;
  }
  v19 = sub_1487400(a1, a2, (__int64)&v21, a3, a4);
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  return v19;
}
