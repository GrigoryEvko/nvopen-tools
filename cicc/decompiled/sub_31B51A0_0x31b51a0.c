// Function: sub_31B51A0
// Address: 0x31b51a0
//
__int64 __fastcall sub_31B51A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rax
  _OWORD *v6; // rax
  _OWORD *v7; // rbx
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // r8
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  bool v19; // zf
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rbx
  int v24; // eax
  _QWORD *v25; // rcx
  unsigned __int64 v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // r12d
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  _QWORD *v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  _QWORD v36[12]; // [rsp+30h] [rbp-60h] BYREF

  v5 = sub_318B4F0(**(_QWORD **)(a2 + 48));
  v32 = sub_371B480(v5);
  v6 = (_OWORD *)sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
  {
    *v6 = 0;
    v6[1] = 0;
    v6[2] = 0;
    v6[3] = 0;
    v6[4] = 0;
  }
  v8 = *(_QWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 88) = v6;
  if ( v8 )
  {
    v9 = *(unsigned int *)(v8 + 56);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD *)(v8 + 40);
      v11 = v10 + 40 * v9;
      do
      {
        if ( *(_QWORD *)v10 != -4096 && *(_QWORD *)v10 != -8192 )
          sub_C7D6A0(*(_QWORD *)(v10 + 16), 16LL * *(unsigned int *)(v10 + 32), 8);
        v10 += 40;
      }
      while ( v11 != v10 );
      v9 = *(unsigned int *)(v8 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(v8 + 40), 40 * v9, 8);
    sub_C7D6A0(*(_QWORD *)(v8 + 8), 16LL * *(unsigned int *)(v8 + 24), 8);
    j_j___libc_free_0(v8);
    v7 = *(_OWORD **)(a1 + 88);
  }
  v12 = *(_QWORD *)(v32 + 24);
  v31 = *(_QWORD *)sub_3187030(v12, *(_QWORD *)(*(_QWORD *)(v32 + 16) + 40LL)) + 312LL;
  v13 = a3[1];
  v14 = *a3;
  v33 = v13;
  v15 = (_QWORD *)sub_22077B0(0x168u);
  v17 = (unsigned __int64)v15;
  if ( v15 )
  {
    v15[1] = 0;
    v15[2] = 0;
    v15[3] = 0;
    sub_31B1840(0, 0);
    sub_31B0790(v17 + 40, v14, v12);
    *(_QWORD *)(v17 + 168) = v12;
    *(_BYTE *)(v17 + 208) = 0;
    *(_QWORD *)(v17 + 216) = 0;
    *(_QWORD *)(v17 + 224) = 0;
    *(_QWORD *)(v17 + 232) = 0;
    *(_DWORD *)(v17 + 240) = 0;
    *(_QWORD *)(v17 + 248) = 0;
    *(_BYTE *)(v17 + 264) = 0;
    v36[1] = sub_31AF830;
    v34 = (_QWORD *)v17;
    v36[0] = sub_31AF900;
    v18 = sub_318AEA0(v12, (__int64)&v34);
    v19 = *(_BYTE *)(v17 + 264) == 0;
    *(_QWORD *)(v17 + 256) = v18;
    if ( v19 )
      *(_BYTE *)(v17 + 264) = 1;
    if ( v36[0] )
      ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v36[0])(&v34, &v34, 3);
    *(_QWORD *)(v17 + 352) = v7;
    *(_QWORD *)(v17 + 272) = v17 + 288;
    *(_QWORD *)(v17 + 280) = 0x600000000LL;
    *(_QWORD *)(v17 + 336) = v33;
    *(_QWORD *)(v17 + 344) = v31;
  }
  v20 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v17;
  if ( v20 )
    sub_31B0BF0(v20);
  v34 = v36;
  v21 = *(unsigned int *)(a2 + 56);
  v22 = *(_QWORD *)(a2 + 48);
  v35 = 0x600000000LL;
  v23 = 8 * v21;
  v24 = v21;
  if ( v21 > 6 )
  {
    sub_C8D5F0((__int64)&v34, v36, v21, 8u, v16, (__int64)&v34);
    v25 = &v34[(unsigned int)v35];
  }
  else
  {
    if ( !v23 )
    {
      v28 = v21;
      v27 = v36;
      goto LABEL_26;
    }
    v25 = v36;
  }
  v26 = 0;
  do
  {
    v25[v26 / 8] = *(_QWORD *)(v22 + v26);
    v26 += 8LL;
  }
  while ( v23 != v26 );
  v27 = v34;
  v24 = v21 + v35;
  v28 = (unsigned int)(v21 + v35);
LABEL_26:
  LODWORD(v35) = v24;
  v29 = sub_31B4ED0(a1, (unsigned __int64)v27, v28);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  return v29;
}
