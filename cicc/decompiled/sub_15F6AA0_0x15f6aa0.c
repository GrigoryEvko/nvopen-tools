// Function: sub_15F6AA0
// Address: 0x15f6aa0
//
__int64 __fastcall sub_15F6AA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  char *v12; // r14
  __int64 v13; // rax
  char *v14; // r15
  __int64 v15; // rax
  __int64 *v16; // r13
  __int64 *v17; // rdx
  __int64 *v18; // rax
  int v19; // r15d
  __int64 v20; // rax
  int v21; // r8d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // rdx
  __int64 v25; // r14
  __int64 *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r15
  _QWORD *v30; // r14
  __int64 v31; // rsi
  __int64 v33; // rsi
  __int64 v34; // [rsp+0h] [rbp-B0h]
  __int64 v35; // [rsp+10h] [rbp-A0h]
  __int64 v37; // [rsp+20h] [rbp-90h]
  __int64 v38; // [rsp+28h] [rbp-88h]
  __int64 v39; // [rsp+30h] [rbp-80h]
  unsigned int v40; // [rsp+40h] [rbp-70h]
  _QWORD v41[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v42[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v43; // [rsp+70h] [rbp-40h]

  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_8;
  v5 = sub_1648A40(a1);
  v7 = v5 + v6;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v7 >> 4) )
LABEL_39:
      BUG();
LABEL_8:
    v11 = -72;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v7 - sub_1648A40(a1)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_39;
  v8 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v9 = sub_1648A40(a1);
  v11 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
LABEL_9:
  v12 = (char *)(a1 + v11);
  v13 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v14 = (char *)(a1 - v13);
  v15 = v11 + v13;
  if ( v15 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v16 = 0;
  v35 = 0x5555555555555558LL * (v15 >> 3);
  if ( 0xAAAAAAAAAAAAAAABLL * (v15 >> 3) )
    v16 = (__int64 *)sub_22077B0(0x5555555555555558LL * (v15 >> 3));
  if ( v14 == v12 )
  {
    v34 = 0;
    v19 = 3;
  }
  else
  {
    v17 = v16;
    v18 = (__int64 *)v14;
    do
    {
      if ( v17 )
        *v17 = *v18;
      v18 += 3;
      ++v17;
    }
    while ( v12 != (char *)v18 );
    v34 = (__int64)(0x5555555555555558LL * ((unsigned __int64)(v12 - v14 - 24) >> 3) + 8) >> 3;
    v19 = v34 + 3;
  }
  v20 = sub_1649960(a1);
  v21 = 0;
  v41[0] = v20;
  v43 = 261;
  v42[0] = v41;
  v22 = *(_QWORD *)(a1 - 24);
  v41[1] = v23;
  v24 = a2;
  v37 = v22;
  v38 = *(_QWORD *)(a1 - 48);
  v39 = *(_QWORD *)(a1 - 72);
  v25 = *(_QWORD *)(*(_QWORD *)v39 + 24LL);
  v26 = &a2[7 * a3];
  if ( v26 != a2 )
  {
    do
    {
      v27 = v24[5] - v24[4];
      v24 += 7;
      v21 += v27 >> 3;
    }
    while ( v26 != v24 );
  }
  v40 = v19 + v21;
  v28 = sub_1648AB0(72, (unsigned int)(v19 + v21), (unsigned int)(16 * a3));
  v29 = v28;
  if ( v28 )
  {
    sub_15F1EA0(v28, **(_QWORD **)(v25 + 16), 5, v28 - 24LL * v40, v40, a4);
    *(_QWORD *)(v29 + 56) = 0;
    sub_15F6500(v29, v25, v39, v38, v37, (__int64)v42, v16, v34, a2, a3);
  }
  v30 = (_QWORD *)(v29 + 48);
  *(_WORD *)(v29 + 18) = *(_WORD *)(v29 + 18) & 0x8000
                       | *(_WORD *)(v29 + 18) & 3
                       | (4 * ((*(_WORD *)(a1 + 18) >> 2) & 0xDFFF));
  *(_BYTE *)(v29 + 17) = *(_BYTE *)(a1 + 17) & 0xFE | *(_BYTE *)(v29 + 17) & 1;
  *(_QWORD *)(v29 + 56) = *(_QWORD *)(a1 + 56);
  v31 = *(_QWORD *)(a1 + 48);
  v42[0] = v31;
  if ( !v31 )
  {
    if ( v30 == v42 || !*(_QWORD *)(v29 + 48) )
      goto LABEL_26;
LABEL_30:
    sub_161E7C0(v29 + 48);
    goto LABEL_31;
  }
  sub_1623A60(v42, v31, 2);
  if ( v30 == v42 )
  {
    if ( v42[0] )
      sub_161E7C0(v42);
    goto LABEL_26;
  }
  if ( *(_QWORD *)(v29 + 48) )
    goto LABEL_30;
LABEL_31:
  v33 = v42[0];
  *(_QWORD *)(v29 + 48) = v42[0];
  if ( v33 )
    sub_1623210(v42, v33, v29 + 48);
LABEL_26:
  if ( v16 )
    j_j___libc_free_0(v16, v35);
  return v29;
}
