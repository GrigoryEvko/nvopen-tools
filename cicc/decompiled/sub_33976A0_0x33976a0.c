// Function: sub_33976A0
// Address: 0x33976a0
//
void __fastcall sub_33976A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned int v6; // edx
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // ecx
  int v15; // r9d
  int v16; // r8d
  __int64 v17; // rax
  _BYTE **v18; // rdx
  __int64 v19; // r15
  _BYTE *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rbx
  int v23; // edx
  int v24; // r12d
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  int v28; // edx
  int v29; // r13d
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  __int64 *v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h] BYREF
  int v35; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v34 = 0;
  v5 = v4;
  v7 = v6;
  v8 = *(_QWORD *)a1;
  v35 = *(_DWORD *)(a1 + 848);
  if ( v8 )
  {
    if ( &v34 != (__int64 *)(v8 + 48) )
    {
      v9 = *(_QWORD *)(v8 + 48);
      v34 = v9;
      if ( v9 )
        sub_B96E90((__int64)&v34, v9, 1);
    }
  }
  v10 = *(_QWORD *)(a1 + 864);
  v11 = *(_QWORD *)(v10 + 16);
  v32 = *(__int64 **)(a2 + 8);
  v12 = sub_2E79000(*(__int64 **)(v10 + 40));
  v14 = sub_2D5BAE0(v11, v12, v32, 0);
  v16 = v13;
  v17 = *(_QWORD *)(v5 + 48) + 16LL * v7;
  if ( (_WORD)v14 != *(_WORD *)v17 || !(_WORD)v14 && *(_QWORD *)(v17 + 8) != v13 )
  {
    v26 = sub_33FAF80(*(_QWORD *)(a1 + 864), 234, (unsigned int)&v34, v14, v13, v15);
    v33 = a2;
    v27 = v26;
    v29 = v28;
    v30 = sub_337DC20(a1 + 8, &v33);
    *v30 = v27;
    *((_DWORD *)v30 + 2) = v29;
    goto LABEL_14;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v18 = *(_BYTE ***)(a2 - 8);
    v19 = a1 + 8;
    v20 = *v18;
    if ( **v18 == 17 )
    {
LABEL_11:
      v21 = sub_34007B0(*(_QWORD *)(a1 + 864), (int)v20 + 24, (unsigned int)&v34, v14, v16, 0, 1);
      v33 = a2;
      v22 = v21;
      v24 = v23;
      v25 = sub_337DC20(v19, &v33);
      *v25 = v22;
      *((_DWORD *)v25 + 2) = v24;
      goto LABEL_14;
    }
  }
  else
  {
    v19 = a1 + 8;
    v20 = *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v20 == 17 )
      goto LABEL_11;
  }
  v33 = a2;
  v31 = sub_337DC20(v19, &v33);
  *v31 = v5;
  *((_DWORD *)v31 + 2) = v7;
LABEL_14:
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
}
