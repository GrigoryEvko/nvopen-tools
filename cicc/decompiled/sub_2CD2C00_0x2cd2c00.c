// Function: sub_2CD2C00
// Address: 0x2cd2c00
//
void __fastcall sub_2CD2C00(_BYTE *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 *v7; // rdx
  unsigned __int64 v8; // r14
  char v9; // bl
  __int64 v10; // rdx
  __int64 *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  int v14; // edx
  int v15; // r13d
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 **v18; // rax
  __int64 **v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // [rsp+0h] [rbp-160h]
  __int64 v22; // [rsp+8h] [rbp-158h]
  __int64 v23; // [rsp+10h] [rbp-150h]
  char v25; // [rsp+28h] [rbp-138h]
  unsigned __int64 v26; // [rsp+28h] [rbp-138h]
  unsigned __int64 v27; // [rsp+38h] [rbp-128h] BYREF
  int v28[8]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v29; // [rsp+60h] [rbp-100h]
  _QWORD v30[2]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v31[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v32; // [rsp+90h] [rbp-D0h]
  unsigned int *v33[2]; // [rsp+A0h] [rbp-C0h] BYREF
  char v34; // [rsp+B0h] [rbp-B0h] BYREF
  void *v35; // [rsp+120h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(unsigned __int64 **)(a2 - 8);
  else
    v7 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *v7;
  v9 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  v25 = *(_BYTE *)(*(_QWORD *)(*v7 + 8) + 8LL);
  if ( v9 == 5 || *(_BYTE *)(*(_QWORD *)(*v7 + 8) + 8LL) == 5 )
  {
    v21 = (_QWORD *)sub_BD5C60(a2);
    v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
    sub_23D0AB0((__int64)v33, a2, 0, 0, 0);
    if ( v25 == 5 )
      v10 = sub_BCB2F0(v21);
    else
      v10 = *(_QWORD *)(v8 + 8);
    if ( v9 == 5 )
    {
      v22 = v10;
      v20 = sub_BCB2F0(v21);
      v10 = v22;
      v11 = (__int64 *)v20;
    }
    else
    {
      v11 = *(__int64 **)(a2 + 8);
    }
    v31[0] = v10;
    v30[0] = v31;
    v30[1] = 0x100000001LL;
    v12 = sub_BCF480(v11, v31, 1, 0);
    v13 = sub_BA8C10(v23, a3, a4, v12, 0);
    v15 = v14;
    v32 = 257;
    if ( v25 == 5 )
    {
      v29 = 257;
      v19 = (__int64 **)sub_BCB2F0(v21);
      v8 = sub_2CD24F0((__int64 *)v33, 0x31u, v8, v19, (__int64)v28, 0, v27, 0);
    }
    v27 = v8;
    v16 = sub_921880(v33, v13, v15, (int)&v27, 1, (__int64)v30, 0);
    v17 = v16;
    if ( v9 == 5 )
    {
      v26 = v16;
      v32 = 257;
      v18 = (__int64 **)sub_BCB1B0(v21);
      v17 = sub_2CD24F0((__int64 *)v33, 0x31u, v26, v18, (__int64)v30, 0, v28[0], 0);
    }
    sub_BD84D0(a2, v17);
    sub_B43D60((_QWORD *)a2);
    *a1 = 1;
    nullsub_61();
    v35 = &unk_49DA100;
    nullsub_63();
    if ( (char *)v33[0] != &v34 )
      _libc_free((unsigned __int64)v33[0]);
  }
}
