// Function: sub_94BE50
// Address: 0x94be50
//
__int64 __fastcall sub_94BE50(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r12d
  __int64 *v5; // rdi
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v10; // rdi
  __int64 v11; // r10
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned int *v17; // rax
  unsigned int *v18; // r13
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v25; // [rsp+8h] [rbp-98h]
  char v26[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v27; // [rsp+30h] [rbp-70h]
  _BYTE v28[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v29; // [rsp+60h] [rbp-40h]

  v2 = a1 + 48;
  v3 = a2;
  v5 = *(__int64 **)(a1 + 32);
  v29 = 257;
  v6 = sub_90A810(v5, a2, 0, 0);
  v7 = 0;
  if ( v6 )
    v7 = *(_QWORD *)(v6 + 24);
  v8 = sub_921880((unsigned int **)v2, v7, v6, 0, 0, (__int64)v28, 0);
  if ( v3 == 8925 )
  {
    v10 = *(_QWORD *)(a1 + 120);
    v27 = 257;
    v11 = sub_BCB2D0(v10);
    if ( v11 == *(_QWORD *)(v8 + 8) )
      return v8;
    v12 = *(_QWORD *)(a1 + 128);
    v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v12 + 120LL);
    if ( v13 == sub_920130 )
    {
      if ( *(_BYTE *)v8 > 0x15u )
        goto LABEL_13;
      v23 = v11;
      if ( (unsigned __int8)sub_AC4810(39) )
        v14 = sub_ADAB70(39, v8, v23, 0);
      else
        v14 = sub_AA93C0(39, v8, v23);
      v11 = v23;
      v15 = v14;
    }
    else
    {
      v25 = v11;
      v22 = v13(v12, 39u, (_BYTE *)v8, v11);
      v11 = v25;
      v15 = v22;
    }
    if ( v15 )
      return v15;
LABEL_13:
    v24 = v11;
    v29 = 257;
    v16 = sub_BD2C40(72, unk_3F10A14);
    v15 = v16;
    if ( v16 )
      sub_B515B0(v16, v8, v24, v28, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      v15,
      v26,
      *(_QWORD *)(v2 + 56),
      *(_QWORD *)(v2 + 64));
    v17 = *(unsigned int **)(a1 + 48);
    v18 = &v17[4 * *(unsigned int *)(a1 + 56)];
    if ( v17 != v18 )
    {
      v19 = *(unsigned int **)(a1 + 48);
      do
      {
        v20 = *((_QWORD *)v19 + 1);
        v21 = *v19;
        v19 += 4;
        sub_B99FD0(v15, v21, v20);
      }
      while ( v18 != v19 );
    }
    return v15;
  }
  return v8;
}
