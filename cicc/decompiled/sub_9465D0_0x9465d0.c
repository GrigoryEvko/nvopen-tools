// Function: sub_9465D0
// Address: 0x9465d0
//
__int64 __fastcall sub_9465D0(__int64 a1, __int64 a2, char a3)
{
  __int64 i; // rax
  __int64 v5; // r12
  __int64 v6; // r15
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v20; // rdx
  unsigned int v21; // r15d
  __int64 v22; // rdx
  unsigned int *v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // [rsp-10h] [rbp-C0h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  unsigned int *v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  _QWORD v33[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v34; // [rsp+40h] [rbp-70h]
  _QWORD v35[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v36; // [rsp+70h] [rbp-40h]

  for ( i = *(_QWORD *)(a2 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(i + 128);
  v6 = sub_9439D0(a1, a2);
  v29 = sub_ACD640(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 704LL), v5, 0);
  v7 = *(_QWORD **)(a1 + 32);
  v34 = 257;
  v8 = *(_QWORD *)(v6 + 8);
  v9 = v7[90];
  if ( v9 == v8 )
  {
    v13 = v6;
    goto LABEL_11;
  }
  v10 = *(_QWORD *)(a1 + 128);
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v11 != sub_920130 )
  {
    v32 = v9;
    v26 = v11(v10, 49u, (_BYTE *)v6, v9);
    v9 = v32;
    v13 = v26;
    goto LABEL_9;
  }
  if ( *(_BYTE *)v6 <= 0x15u )
  {
    v30 = v9;
    if ( (unsigned __int8)sub_AC4810(49) )
      v12 = sub_ADAB70(49, v6, v30, 0);
    else
      v12 = sub_AA93C0(49, v6, v30);
    v9 = v30;
    v13 = v12;
LABEL_9:
    if ( v13 )
    {
      v7 = *(_QWORD **)(a1 + 32);
      v8 = v7[90];
      goto LABEL_11;
    }
  }
  v36 = 257;
  v13 = sub_B51D30(49, v6, v9, v35, 0, 0);
  if ( (unsigned __int8)sub_920620(v13) )
  {
    v20 = *(_QWORD *)(a1 + 144);
    v21 = *(_DWORD *)(a1 + 152);
    if ( v20 )
      sub_B99FD0(v13, 3, v20);
    sub_B45150(v13, v21);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v13,
    v33,
    *(_QWORD *)(a1 + 104),
    *(_QWORD *)(a1 + 112));
  v22 = 4LL * *(unsigned int *)(a1 + 56);
  v23 = *(unsigned int **)(a1 + 48);
  v31 = &v23[v22];
  while ( v31 != v23 )
  {
    v24 = *((_QWORD *)v23 + 1);
    v25 = *v23;
    v23 += 4;
    sub_B99FD0(v13, v25, v24);
  }
  v7 = *(_QWORD **)(a1 + 32);
  v8 = v7[90];
LABEL_11:
  v35[0] = v8;
  if ( a3 )
    v14 = sub_B6E160(*v7, 210, v35, 1);
  else
    v14 = sub_B6E160(*v7, 211, v35, 1);
  v33[1] = v13;
  v15 = 0;
  v36 = 257;
  v33[0] = v29;
  if ( v14 )
    v15 = *(_QWORD *)(v14 + 24);
  v16 = sub_921880((unsigned int **)(a1 + 48), v15, v14, (int)v33, 2, (__int64)v35, 0);
  v18 = sub_BD5C60(v16, v15, v17);
  *(_QWORD *)(v16 + 72) = sub_A7A090(v16 + 72, v18, 0xFFFFFFFFLL, 41);
  return v27;
}
