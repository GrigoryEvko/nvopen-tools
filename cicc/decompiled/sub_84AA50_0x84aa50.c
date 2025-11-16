// Function: sub_84AA50
// Address: 0x84aa50
//
_BOOL8 __fastcall sub_84AA50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        const __m128i *a7)
{
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // r8
  unsigned __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // r8
  __int64 *v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 *v26; // r9
  _QWORD *v27; // r12
  _BOOL4 i; // r14d
  _QWORD *v29; // rbx
  unsigned int v32; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  int v36; // [rsp+20h] [rbp-40h] BYREF
  int v37; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a3;
  v38[0] = 0;
  v36 = 0;
  v37 = 0;
  v8 = sub_82BD70(a1, a2, a3);
  v10 = a6;
  v11 = a5;
  v12 = *(_QWORD *)(v8 + 1024);
  v13 = a4;
  v14 = v8;
  if ( v12 == *(_QWORD *)(v8 + 1016) )
  {
    v32 = a6;
    v35 = v13;
    sub_8332F0(v8, a2, v13, a5, v9, (__int64 *)v10);
    LODWORD(v10) = v32;
    v11 = a5;
    v13 = v35;
  }
  v15 = *(_QWORD *)(v14 + 1008) + 40 * v12;
  if ( v15 )
  {
    *(_BYTE *)v15 &= 0xFCu;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)(v15 + 24) = 0;
    *(_QWORD *)(v15 + 32) = 0;
  }
  *(_QWORD *)(v14 + 1024) = v12 + 1;
  sub_8360D0(a1, v7, v13, v11, 0, v10, a7, 0, 0, 1u, 0, 0, 0, 0, 0, 0, a2, v38, 0, &v36, &v37);
  v18 = sub_82BD70(a1, v7, v16);
  v21 = *(_QWORD *)(v18 + 1008);
  v22 = *(_QWORD *)(v21 + 8 * (5LL * *(_QWORD *)(v18 + 1024) - 5) + 32);
  if ( v22 )
  {
    sub_823A00(*(_QWORD *)v22, 16LL * (unsigned int)(*(_DWORD *)(v22 + 8) + 1), v21, v17, v19, v20);
    sub_823A00(v22, 16, v23, v24, v25, v26);
  }
  v27 = (_QWORD *)v38[0];
  --*(_QWORD *)(v18 + 1024);
  for ( i = v27 != 0; v27; qword_4D03C68 = v29 )
  {
    v29 = v27;
    v27 = (_QWORD *)*v27;
    sub_725130((__int64 *)v29[5]);
    sub_82D8A0((_QWORD *)v29[15]);
    *v29 = qword_4D03C68;
  }
  return i;
}
