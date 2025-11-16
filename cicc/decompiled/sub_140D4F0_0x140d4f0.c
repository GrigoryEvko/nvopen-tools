// Function: sub_140D4F0
// Address: 0x140d4f0
//
_QWORD *__fastcall sub_140D4F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  unsigned __int64 v3; // r12
  int v6; // edx
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // r15
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // [rsp+18h] [rbp-98h] BYREF
  __m128i v34; // [rsp+20h] [rbp-90h] BYREF
  char v35; // [rsp+30h] [rbp-80h]
  _BYTE v36[16]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v37; // [rsp+50h] [rbp-60h]
  _BYTE v38[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v39; // [rsp+70h] [rbp-40h]

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = v2;
  sub_140AD80(&v34, v2, *(_QWORD **)(a1 + 8));
  if ( !v35 || v34.m128i_i8[0] == 16 )
    return 0;
  v6 = *(_DWORD *)(v2 + 20);
  v37 = 257;
  v7 = v34.m128i_u32[2] - (unsigned __int64)(v6 & 0xFFFFFFF);
  v8 = *(_QWORD *)(a1 + 96);
  v9 = 3 * v7;
  v10 = *(_QWORD *)(v2 + 8 * v9);
  if ( v8 != *(_QWORD *)v10 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      v18 = *(_QWORD *)(v2 + 8 * v9);
      v39 = 257;
      v19 = sub_15FDBD0(37, v18, v8, v38, 0);
      v20 = *(_QWORD *)(a1 + 32);
      v10 = v19;
      if ( v20 )
      {
        v21 = *(__int64 **)(a1 + 40);
        sub_157E9D0(v20 + 40, v19);
        v22 = *(_QWORD *)(v10 + 24);
        v23 = *v21;
        *(_QWORD *)(v10 + 32) = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v10 + 24;
        *v21 = *v21 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v36);
      v24 = *(_QWORD *)(a1 + 24);
      if ( v24 )
      {
        v33 = *(_QWORD *)(a1 + 24);
        sub_1623A60(&v33, v24, 2);
        if ( *(_QWORD *)(v10 + 48) )
          sub_161E7C0(v10 + 48);
        v25 = v33;
        *(_QWORD *)(v10 + 48) = v33;
        if ( v25 )
          sub_1623210(&v33, v25, v10 + 48);
      }
    }
    else
    {
      v10 = sub_15A46C0(37, *(_QWORD *)(v2 + 8 * v9), v8, 0);
      v11 = sub_14DBA30(v10, *(_QWORD *)(a1 + 88), 0);
      if ( v11 )
        v10 = v11;
    }
  }
  if ( v34.m128i_i32[3] < 0 )
    return (_QWORD *)v10;
  v12 = *(_DWORD *)(v3 + 20);
  v37 = 257;
  v13 = v34.m128i_i32[3] - (unsigned __int64)(v12 & 0xFFFFFFF);
  v14 = *(_QWORD *)(a1 + 96);
  v15 = *(_QWORD *)(v3 + 24 * v13);
  if ( v14 != *(_QWORD *)v15 )
  {
    if ( *(_BYTE *)(v15 + 16) > 0x10u )
    {
      v39 = 257;
      v15 = sub_15FDBD0(37, v15, v14, v38, 0);
      v26 = *(_QWORD *)(a1 + 32);
      if ( v26 )
      {
        v27 = *(__int64 **)(a1 + 40);
        sub_157E9D0(v26 + 40, v15);
        v28 = *(_QWORD *)(v15 + 24);
        v29 = *v27;
        *(_QWORD *)(v15 + 32) = v27;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v15 + 24) = v29 | v28 & 7;
        *(_QWORD *)(v29 + 8) = v15 + 24;
        *v27 = *v27 & 7 | (v15 + 24);
      }
      sub_164B780(v15, v36);
      v30 = *(_QWORD *)(a1 + 24);
      if ( v30 )
      {
        v33 = *(_QWORD *)(a1 + 24);
        sub_1623A60(&v33, v30, 2);
        v31 = v15 + 48;
        if ( *(_QWORD *)(v15 + 48) )
        {
          sub_161E7C0(v15 + 48);
          v31 = v15 + 48;
        }
        v32 = v33;
        *(_QWORD *)(v15 + 48) = v33;
        if ( v32 )
          sub_1623210(&v33, v32, v31);
      }
    }
    else
    {
      v16 = sub_15A46C0(37, v15, v14, 0);
      v17 = sub_14DBA30(v16, *(_QWORD *)(a1 + 88), 0);
      if ( v17 )
        v16 = v17;
      v15 = v16;
    }
  }
  v39 = 257;
  return sub_140D0B0((__int64 *)(a1 + 24), v10, v15, (__int64)v38, 0, 0);
}
