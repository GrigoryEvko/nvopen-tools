// Function: sub_927810
// Address: 0x927810
//
__int64 __fastcall sub_927810(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r13
  __int64 *v5; // rdi
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v14; // rdx
  unsigned int v15; // r12d
  unsigned int *v16; // rax
  unsigned int *v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int *v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v23[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v24; // [rsp+40h] [rbp-70h]
  _BYTE v25[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v26; // [rsp+70h] [rbp-40h]

  v3 = a1 + 48;
  v5 = *(__int64 **)(a1 + 32);
  v24 = 257;
  v22 = a2;
  v6 = v5[87];
  if ( v6 == *(_QWORD *)(a2 + 8) )
  {
    v9 = a2;
    goto LABEL_8;
  }
  v7 = *(_QWORD *)(a1 + 128);
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v7 + 120LL);
  if ( v8 != sub_920130 )
  {
    v9 = v8(v7, 49u, (_BYTE *)a2, v6);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(49) )
      v9 = sub_ADAB70(49, a2, v6, 0);
    else
      v9 = sub_AA93C0(49, a2, v6);
LABEL_6:
    if ( v9 )
    {
      v5 = *(__int64 **)(a1 + 32);
      goto LABEL_8;
    }
  }
  v26 = 257;
  v9 = sub_B51D30(49, a2, v6, v25, 0, 0);
  if ( (unsigned __int8)sub_920620(v9) )
  {
    v14 = *(_QWORD *)(a1 + 144);
    v15 = *(_DWORD *)(a1 + 152);
    if ( v14 )
      sub_B99FD0(v9, 3, v14);
    sub_B45150(v9, v15);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v9,
    v23,
    *(_QWORD *)(v3 + 56),
    *(_QWORD *)(v3 + 64));
  v16 = *(unsigned int **)(a1 + 48);
  v17 = v16;
  v21 = &v16[4 * *(unsigned int *)(a1 + 56)];
  if ( v16 != v21 )
  {
    do
    {
      v18 = *((_QWORD *)v17 + 1);
      v19 = *v17;
      v17 += 4;
      sub_B99FD0(v9, v19, v18);
    }
    while ( v21 != v17 );
  }
  v5 = *(__int64 **)(a1 + 32);
LABEL_8:
  v10 = *(_QWORD *)(v9 + 8);
  v22 = v9;
  v23[0] = v10;
  v26 = 257;
  v11 = sub_90A810(v5, 374 - ((unsigned int)(a3 == 0) - 1), (__int64)v23, 1u);
  v12 = 0;
  if ( v11 )
    v12 = *(_QWORD *)(v11 + 24);
  return sub_921880((unsigned int **)v3, v12, v11, (int)&v22, 1, (__int64)v25, 0);
}
