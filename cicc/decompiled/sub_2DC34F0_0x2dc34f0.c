// Function: sub_2DC34F0
// Address: 0x2dc34f0
//
_QWORD *__fastcall sub_2DC34F0(_QWORD *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // r13d
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int8 *v9; // r10
  __int64 v10; // r15
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int8 *v14; // r14
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rax
  _QWORD *v23; // rax
  _BYTE *v24; // r8
  unsigned __int8 *v25; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v26; // [rsp+8h] [rbp-A8h]
  char v29[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v30; // [rsp+40h] [rbp-70h]
  _QWORD v31[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v32; // [rsp+70h] [rbp-40h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v4 = *a3;
  v5 = a3[1];
  if ( v5 - *a3 == 8 )
  {
    v24 = 0;
    goto LABEL_28;
  }
  v6 = 0;
  v7 = 0;
  do
  {
    v30 = 257;
    v9 = *(unsigned __int8 **)(v4 + 8 * v7);
    v13 = *a2;
    v14 = *(unsigned __int8 **)(v4 + 8LL * (v6 + 1));
    v15 = *(_QWORD *)(*a2 + 208);
    v16 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v15 + 16LL);
    if ( v16 == sub_9202E0 )
    {
      if ( *v9 > 0x15u || *v14 > 0x15u )
        goto LABEL_15;
      v25 = v9;
      if ( (unsigned __int8)sub_AC47B0(29) )
        v8 = sub_AD5570(29, (__int64)v25, v14, 0, 0);
      else
        v8 = sub_AABE40(0x1Du, v25, v14);
      v9 = v25;
      v10 = v8;
    }
    else
    {
      v26 = v9;
      v21 = v16(v15, 29u, v9, v14);
      v9 = v26;
      v10 = v21;
    }
    if ( v10 )
      goto LABEL_8;
LABEL_15:
    v32 = 257;
    v10 = sub_B504D0(29, (__int64)v9, (__int64)v14, (__int64)v31, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v13 + 216) + 16LL))(
      *(_QWORD *)(v13 + 216),
      v10,
      v29,
      *(_QWORD *)(v13 + 184),
      *(_QWORD *)(v13 + 192));
    v17 = *(_QWORD *)(v13 + 128);
    v18 = v17 + 16LL * *(unsigned int *)(v13 + 136);
    if ( v17 == v18 )
    {
LABEL_8:
      v31[0] = v10;
      v11 = (_BYTE *)a1[1];
      if ( v11 == (_BYTE *)a1[2] )
        goto LABEL_18;
      goto LABEL_9;
    }
    do
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v10, v20, v19);
    }
    while ( v18 != v17 );
    v31[0] = v10;
    v11 = (_BYTE *)a1[1];
    if ( v11 == (_BYTE *)a1[2] )
    {
LABEL_18:
      sub_9281F0((__int64)a1, v11, v31);
      goto LABEL_12;
    }
LABEL_9:
    if ( v11 )
    {
      *(_QWORD *)v11 = v10;
      v11 = (_BYTE *)a1[1];
    }
    a1[1] = v11 + 8;
LABEL_12:
    v6 += 2;
    v7 = v6;
    v5 = a3[1];
    v4 = *a3;
    v12 = (v5 - *a3) >> 3;
  }
  while ( v6 < (unsigned __int64)(v12 - 1) );
  if ( (v12 & 1) == 0 )
    return a1;
  v23 = (_QWORD *)a1[1];
  v24 = (_BYTE *)a1[2];
  if ( v23 == (_QWORD *)v24 )
  {
LABEL_28:
    sub_9281F0((__int64)a1, v24, (_QWORD *)(v5 - 8));
    return a1;
  }
  if ( v23 )
  {
    *v23 = *(_QWORD *)(v5 - 8);
    v23 = (_QWORD *)a1[1];
  }
  a1[1] = v23 + 1;
  return a1;
}
