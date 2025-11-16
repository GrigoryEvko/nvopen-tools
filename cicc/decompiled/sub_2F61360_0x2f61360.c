// Function: sub_2F61360
// Address: 0x2f61360
//
__int64 __fastcall sub_2F61360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  __int64 v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // r9
  _QWORD *v12; // rax
  __int64 result; // rax
  __int64 *v16; // [rsp+10h] [rbp-60h]
  __int64 v17; // [rsp+18h] [rbp-58h]
  _QWORD v18[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v19)(unsigned __int64 *, const __m128i **, int); // [rsp+30h] [rbp-40h]
  __int64 (__fastcall *v20)(); // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 40);
  v10 = *(_QWORD **)(a1 + 24);
  v11 = *(_QWORD *)(v9 + 32);
  v19 = 0;
  v16 = (__int64 *)(v9 + 56);
  v17 = v11;
  v12 = (_QWORD *)sub_22077B0(0x20u);
  if ( v12 )
  {
    *v12 = a1;
    v12[1] = v16;
    v12[2] = a3;
    v12[3] = a6;
  }
  v18[0] = v12;
  v20 = sub_2F686D0;
  v19 = sub_2F61050;
  sub_2E0C490(a2, v16, a4, a5, (unsigned __int64)v18, v17, v10, a7);
  result = (__int64)v19;
  if ( v19 )
    return v19(v18, (const __m128i **)v18, 3);
  return result;
}
