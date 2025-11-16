// Function: sub_2FDD0C0
// Address: 0x2fdd0c0
//
__int64 __fastcall sub_2FDD0C0(__int64 *a1, unsigned int *a2, unsigned __int64 a3, unsigned int a4, __int64 *a5)
{
  unsigned int v5; // ebx
  __int64 v8; // rsi
  __int64 (__fastcall *v9)(__int64, __int64, __int64 *, unsigned __int64); // rax
  __int64 v10; // r13
  unsigned __int16 v11; // r15
  __int16 v12; // ax
  __int64 v13; // rdx
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // r10d
  __int128 v20; // xmm0
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  void *v24; // r9
  unsigned __int8 v26; // [rsp+7h] [rbp-A9h]
  unsigned __int64 v27; // [rsp+8h] [rbp-A8h]
  __m128i v28; // [rsp+20h] [rbp-90h] BYREF
  __int64 v29; // [rsp+30h] [rbp-80h]
  __int128 v30; // [rsp+40h] [rbp-70h]
  __int64 v31; // [rsp+50h] [rbp-60h]
  _QWORD v32[4]; // [rsp+60h] [rbp-50h] BYREF

  if ( a3 > 1 )
    return 0;
  v5 = *a2;
  if ( !sub_2E8E6C0((__int64)a1, *a2) )
    return 0;
  v8 = a1[3];
  v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, unsigned __int64))(*a5 + 248);
  if ( v9 == sub_2FDCAD0 )
    v10 = sub_2E7E840(*(_QWORD **)(v8 + 32), v8, a1, (unsigned __int64)a1);
  else
    v10 = v9((__int64)a5, v8, a1, (unsigned __int64)a1);
  sub_2FDCAE0(v10, v5, a4, a5);
  v11 = 0;
  v12 = sub_2E923D0((unsigned __int64)a1, *(_DWORD *)(a1[4] + 40LL * v5 + 8), 0);
  v13 = *(_QWORD *)(v10 + 32);
  if ( (_BYTE)v12 )
  {
    *(_QWORD *)(v13 + 64) |= 8uLL;
    v11 = 1;
  }
  if ( HIBYTE(v12) )
  {
    *(_QWORD *)(v13 + 64) |= 0x10uLL;
    v11 |= 2u;
  }
  v14 = (_QWORD *)sub_2E88D60(v10);
  v15 = v14[6];
  v16 = *(_DWORD *)(v15 + 32) + a4;
  v17 = *(_QWORD *)(v15 + 8);
  memset(v32, 0, sizeof(v32));
  v18 = v17 + 40 * v16;
  v26 = *(_BYTE *)(v18 + 16);
  v27 = *(_QWORD *)(v18 + 8);
  sub_2EAC300((__int64)&v28, (__int64)v14, a4, 0);
  v19 = v27;
  v31 = v29;
  v20 = (__int128)_mm_loadu_si128(&v28);
  if ( v27 > 0x3FFFFFFFFFFFFFFBLL )
    v19 = -2;
  v30 = v20;
  v21 = sub_2E7BD70(v14, v11, v19, v26, (int)v32, 0, v20, v29, 1u, 0, 0);
  sub_2E86C70(v10, (__int64)v14, v21, v22, v23, v24);
  return v10;
}
