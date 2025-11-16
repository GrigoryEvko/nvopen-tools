// Function: sub_34335B0
// Address: 0x34335b0
//
unsigned __int64 __fastcall sub_34335B0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // ecx
  __int128 v9; // [rsp-30h] [rbp-A0h]
  __m128i v10; // [rsp+0h] [rbp-70h] BYREF
  __int64 v11; // [rsp+10h] [rbp-60h]
  __m128i v12; // [rsp+20h] [rbp-50h]
  __int64 v13; // [rsp+30h] [rbp-40h]
  _QWORD v14[4]; // [rsp+40h] [rbp-30h] BYREF

  sub_2EAC300((__int64)&v10, (__int64)a1, *(_DWORD *)(a2 + 96), 0);
  v2 = a1[6];
  v3 = 5LL * (unsigned int)(*(_DWORD *)(v2 + 32) + *(_DWORD *)(a2 + 96));
  v4 = *(_QWORD *)(v2 + 8);
  v12 = _mm_loadu_si128(&v10);
  v5 = v4 + 8 * v3;
  memset(v14, 0, sizeof(v14));
  v6 = *(_QWORD *)(v5 + 8);
  v7 = *(unsigned __int8 *)(v5 + 16);
  v13 = v11;
  *((_QWORD *)&v9 + 1) = v12.m128i_i64[1];
  if ( v6 > 0x3FFFFFFFFFFFFFFBLL )
    LODWORD(v6) = -2;
  *(_QWORD *)&v9 = v12.m128i_i64[0];
  return sub_2E7BD70(a1, 7u, v6, v7, (int)v14, 0, v9, v11, 1u, 0, 0);
}
