// Function: sub_927A80
// Address: 0x927a80
//
__int64 __fastcall sub_927A80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 *v6; // rdi
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // rdi
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v21; // rdx
  unsigned int v22; // r15d
  __int64 v23; // rdx
  unsigned int *v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned int v28; // r13d
  unsigned int *v29; // rax
  unsigned int *v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // [rsp+8h] [rbp-B8h]
  unsigned int *v36; // [rsp+8h] [rbp-B8h]
  unsigned int *v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  __int64 v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+18h] [rbp-A8h]
  __int64 v41; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v42[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v43; // [rsp+50h] [rbp-70h]
  _BYTE v44[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v45; // [rsp+80h] [rbp-40h]

  v3 = a1 + 48;
  v6 = *(__int64 **)(a1 + 32);
  v43 = 257;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = v6[87];
  if ( v8 == v7 )
  {
    v12 = a2;
LABEL_9:
    v43 = 257;
    if ( v7 != *(_QWORD *)(a3 + 8) )
      goto LABEL_10;
    goto LABEL_27;
  }
  v9 = *(_QWORD *)(a1 + 128);
  v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v9 + 120LL);
  if ( v10 == sub_920130 )
  {
    if ( *(_BYTE *)a2 > 0x15u )
      goto LABEL_20;
    v39 = v8;
    if ( (unsigned __int8)sub_AC4810(49) )
      v11 = sub_ADAB70(49, a2, v39, 0);
    else
      v11 = sub_AA93C0(49, a2, v39);
    v8 = v39;
    v12 = v11;
  }
  else
  {
    v40 = v8;
    v34 = v10(v9, 49u, (_BYTE *)a2, v8);
    v8 = v40;
    v12 = v34;
  }
  if ( v12 )
  {
    v6 = *(__int64 **)(a1 + 32);
    v7 = v6[87];
    goto LABEL_9;
  }
LABEL_20:
  v45 = 257;
  v12 = sub_B51D30(49, a2, v8, v44, 0, 0);
  if ( (unsigned __int8)sub_920620(v12) )
  {
    v21 = *(_QWORD *)(a1 + 144);
    v22 = *(_DWORD *)(a1 + 152);
    if ( v21 )
      sub_B99FD0(v12, 3, v21);
    sub_B45150(v12, v22);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v12,
    v42,
    *(_QWORD *)(v3 + 56),
    *(_QWORD *)(v3 + 64));
  v23 = 4LL * *(unsigned int *)(a1 + 56);
  v24 = *(unsigned int **)(a1 + 48);
  v36 = &v24[v23];
  while ( v36 != v24 )
  {
    v25 = *((_QWORD *)v24 + 1);
    v26 = *v24;
    v24 += 4;
    sub_B99FD0(v12, v26, v25);
  }
  v6 = *(__int64 **)(a1 + 32);
  v7 = v6[87];
  v43 = 257;
  if ( v7 != *(_QWORD *)(a3 + 8) )
  {
LABEL_10:
    v13 = *(_QWORD *)(a1 + 128);
    v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v13 + 120LL);
    if ( v14 == sub_920130 )
    {
      if ( *(_BYTE *)a3 > 0x15u )
        goto LABEL_28;
      v35 = v7;
      if ( (unsigned __int8)sub_AC4810(49) )
        v15 = sub_ADAB70(49, a3, v35, 0);
      else
        v15 = sub_AA93C0(49, a3, v35);
      v7 = v35;
      v16 = v15;
    }
    else
    {
      v38 = v7;
      v33 = v14(v13, 49u, (_BYTE *)a3, v7);
      v7 = v38;
      v16 = v33;
    }
    if ( v16 )
    {
LABEL_16:
      v6 = *(__int64 **)(a1 + 32);
      goto LABEL_17;
    }
LABEL_28:
    v45 = 257;
    v16 = sub_B51D30(49, a3, v7, v44, 0, 0);
    if ( (unsigned __int8)sub_920620(v16) )
    {
      v27 = *(_QWORD *)(a1 + 144);
      v28 = *(_DWORD *)(a1 + 152);
      if ( v27 )
        sub_B99FD0(v16, 3, v27);
      sub_B45150(v16, v28);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      v16,
      v42,
      *(_QWORD *)(v3 + 56),
      *(_QWORD *)(v3 + 64));
    v29 = *(unsigned int **)(a1 + 48);
    v30 = v29;
    v37 = &v29[4 * *(unsigned int *)(a1 + 56)];
    if ( v29 != v37 )
    {
      do
      {
        v31 = *((_QWORD *)v30 + 1);
        v32 = *v30;
        v30 += 4;
        sub_B99FD0(v16, v32, v31);
      }
      while ( v37 != v30 );
    }
    goto LABEL_16;
  }
LABEL_27:
  v16 = a3;
LABEL_17:
  v17 = *(_QWORD *)(v12 + 8);
  v42[0] = v12;
  v41 = v17;
  v42[1] = v16;
  v45 = 257;
  v18 = sub_90A810(v6, 373, (__int64)&v41, 1u);
  v19 = 0;
  if ( v18 )
    v19 = *(_QWORD *)(v18 + 24);
  return sub_921880((unsigned int **)v3, v19, v18, (int)v42, 2, (__int64)v44, 0);
}
