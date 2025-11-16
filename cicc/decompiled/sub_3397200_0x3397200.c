// Function: sub_3397200
// Address: 0x3397200
//
_QWORD *__fastcall sub_3397200(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r12
  unsigned int v22; // edx
  unsigned __int64 v23; // rbx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rsi
  __int64 v28; // r12
  int v29; // edx
  int v30; // ebx
  _QWORD *result; // rax
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  unsigned int v34; // [rsp+18h] [rbp-88h]
  __int64 v35; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+60h] [rbp-40h] BYREF
  int v38; // [rsp+68h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(__int64 **)(a2 - 8);
  else
    v2 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v3 = sub_338B750(a1, *v2);
  v4 = *(_QWORD *)(a1 + 864);
  v6 = v5;
  v7 = *(__int64 **)(a2 + 8);
  v8 = *(_QWORD *)(v4 + 16);
  v9 = sub_2E79000(*(__int64 **)(v4 + 40));
  v34 = sub_2D5BAE0(v8, v9, v7, 0);
  v35 = v10;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v11 = *(_QWORD *)(a2 - 8);
  else
    v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v12 = *(_QWORD *)(*(_QWORD *)v11 + 8LL);
  v13 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v14 = sub_336EEB0(v8, v13, v12, 0);
  v37 = 0;
  v16 = *(_QWORD *)(a1 + 864);
  v17 = v15;
  v18 = v14;
  v19 = *(_QWORD *)a1;
  v38 = *(_DWORD *)(a1 + 848);
  if ( v19 )
  {
    if ( &v37 != (__int64 *)(v19 + 48) )
    {
      v20 = *(_QWORD *)(v19 + 48);
      v37 = v20;
      if ( v20 )
      {
        v32 = v18;
        v33 = v15;
        sub_B96E90((__int64)&v37, v20, 1);
        v18 = v32;
        v17 = v33;
      }
    }
  }
  v21 = sub_33FB4C0(v16, v3, v6, &v37, v18, v17);
  v23 = v22 | v6 & 0xFFFFFFFF00000000LL;
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  v24 = *(_DWORD *)(a1 + 848);
  v25 = *(_QWORD *)a1;
  v37 = 0;
  v26 = *(_QWORD *)(a1 + 864);
  v38 = v24;
  if ( v25 )
  {
    if ( &v37 != (__int64 *)(v25 + 48) )
    {
      v27 = *(_QWORD *)(v25 + 48);
      v37 = v27;
      if ( v27 )
        sub_B96E90((__int64)&v37, v27, 1);
    }
  }
  v28 = sub_33FB310(v26, v21, v23, &v37, v34, v35);
  v30 = v29;
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  v37 = a2;
  result = sub_337DC20(a1 + 8, &v37);
  *result = v28;
  *((_DWORD *)result + 2) = v30;
  return result;
}
