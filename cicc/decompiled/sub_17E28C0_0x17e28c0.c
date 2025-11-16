// Function: sub_17E28C0
// Address: 0x17e28c0
//
void __fastcall sub_17E28C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 v10; // r15
  __int64 *v11; // r14
  __int64 *v12; // rcx
  __int64 *v13; // rdx
  int v14; // esi
  __int64 v15; // rax
  _QWORD *v16; // rax
  int v17; // r8d
  __int64 v18; // r10
  __int64 v19; // rbx
  __int64 *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // r8d
  __int64 v24; // r9
  char v25; // al
  int v26; // r13d
  __int64 v27; // rdi
  __int64 *v28; // r13
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  _QWORD *v34; // rax
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+28h] [rbp-58h]
  int v43; // [rsp+28h] [rbp-58h]
  _QWORD v44[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v45; // [rsp+40h] [rbp-40h]

  v10 = *(_QWORD *)(a1 + 56);
  v11 = *(__int64 **)(a1 + 48);
  v45 = 257;
  v12 = &v11[7 * v10];
  if ( v12 == v11 )
  {
    v43 = a5 + 1;
    v34 = sub_1648AB0(72, (int)a5 + 1, 16 * (int)v10);
    v23 = v43;
    v24 = a5;
    v18 = a2;
    v19 = (__int64)v34;
    if ( v34 )
    {
      v42 = (__int64)v34;
LABEL_8:
      v36 = v18;
      sub_15F1EA0(v19, **(_QWORD **)(v18 + 16), 54, v19 - 24 * v24 - 24, v23, 0);
      *(_QWORD *)(v19 + 56) = 0;
      sub_15F5B40(v19, v36, a3, a4, a5, (__int64)v44, v11, v10);
      goto LABEL_9;
    }
  }
  else
  {
    v13 = v11;
    v14 = 0;
    do
    {
      v15 = v13[5] - v13[4];
      v13 += 7;
      v14 += v15 >> 3;
    }
    while ( v12 != v13 );
    v35 = a2;
    v41 = a5 + 1;
    v16 = sub_1648AB0(72, (int)a5 + 1 + v14, 16 * (int)v10);
    v17 = v41;
    v18 = v35;
    v19 = (__int64)v16;
    if ( v16 )
    {
      v42 = (__int64)v16;
      v20 = v11;
      LODWORD(v21) = 0;
      do
      {
        v22 = v20[5] - v20[4];
        v20 += 7;
        v21 = (unsigned int)(v22 >> 3) + (unsigned int)v21;
      }
      while ( &v11[7 * v10] != v20 );
      v23 = v21 + v17;
      v24 = a5 + v21;
      goto LABEL_8;
    }
  }
  v42 = 0;
  v19 = 0;
LABEL_9:
  v25 = *(_BYTE *)(*(_QWORD *)v19 + 8LL);
  if ( v25 == 16 )
    v25 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v19 + 16LL) + 8LL);
  if ( (unsigned __int8)(v25 - 1) <= 5u || *(_BYTE *)(v19 + 16) == 76 )
  {
    v26 = *(_DWORD *)(a1 + 40);
    if ( a7 || (a7 = *(_QWORD *)(a1 + 32)) != 0 )
      sub_1625C10(v19, 3, a7);
    sub_15F2440(v19, v26);
  }
  v27 = *(_QWORD *)(a1 + 8);
  if ( v27 )
  {
    v28 = *(__int64 **)(a1 + 16);
    sub_157E9D0(v27 + 40, v19);
    v29 = *(_QWORD *)(v19 + 24);
    v30 = *v28;
    *(_QWORD *)(v19 + 32) = v28;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v19 + 24) = v30 | v29 & 7;
    *(_QWORD *)(v30 + 8) = v19 + 24;
    *v28 = *v28 & 7 | (v19 + 24);
  }
  sub_164B780(v42, a6);
  v31 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v44[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)v44, v31, 2);
    v32 = *(_QWORD *)(v19 + 48);
    if ( v32 )
      sub_161E7C0(v19 + 48, v32);
    v33 = (unsigned __int8 *)v44[0];
    *(_QWORD *)(v19 + 48) = v44[0];
    if ( v33 )
      sub_1623210((__int64)v44, v33, v19 + 48);
  }
}
