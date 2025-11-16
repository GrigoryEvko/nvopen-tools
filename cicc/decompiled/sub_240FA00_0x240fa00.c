// Function: sub_240FA00
// Address: 0x240fa00
//
unsigned __int8 *__fastcall sub_240FA00(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 (__fastcall *v8)(__int64, __int64, __int64); // rax
  unsigned __int8 *v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdx
  unsigned int v13; // esi
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int8 *v20; // r14
  __int64 (__fastcall *v21)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int8 *v26; // r14
  __int64 (__fastcall *v27)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v28; // r9
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // rbx
  __int64 v34; // r14
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  _BYTE v43[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v44; // [rsp+30h] [rbp-70h]
  _BYTE v45[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v46; // [rsp+60h] [rbp-40h]

  v44 = 257;
  v6 = *(_QWORD *)(a1 + 64);
  if ( v6 == *(_QWORD *)(a2 + 8) )
  {
    v9 = (unsigned __int8 *)a2;
  }
  else if ( *(_BYTE *)a2 > 0x15u )
  {
    v46 = 257;
    v9 = (unsigned __int8 *)sub_B52210(a2, v6, (__int64)v45, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v9,
      v43,
      a3[7],
      a3[8]);
    v29 = *a3;
    v30 = *a3 + 16LL * *((unsigned int *)a3 + 2);
    if ( *a3 != v30 )
    {
      do
      {
        v31 = *(_QWORD *)(v29 + 8);
        v32 = *(_DWORD *)v29;
        v29 += 16;
        sub_B99FD0((__int64)v9, v32, v31);
      }
      while ( v30 != v29 );
    }
  }
  else
  {
    v7 = a3[10];
    v8 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v7 + 136LL);
    if ( v8 == sub_928970 )
      v9 = (unsigned __int8 *)sub_ADAFB0(a2, v6);
    else
      v9 = (unsigned __int8 *)v8(v7, a2, *(_QWORD *)(a1 + 64));
    if ( *v9 > 0x1Cu )
    {
      (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
        a3[11],
        v9,
        v43,
        a3[7],
        a3[8]);
      v10 = *a3;
      v11 = *a3 + 16LL * *((unsigned int *)a3 + 2);
      if ( *a3 != v11 )
      {
        do
        {
          v12 = *(_QWORD *)(v10 + 8);
          v13 = *(_DWORD *)v10;
          v10 += 16;
          sub_B99FD0((__int64)v9, v13, v12);
        }
        while ( v11 != v10 );
      }
    }
  }
  v14 = *(_QWORD **)(a1 + 920);
  if ( *v14 )
  {
    v23 = ~*v14;
    v44 = 257;
    v24 = sub_ACD640(*(_QWORD *)(a1 + 64), v23, 0);
    v25 = a3[10];
    v26 = (unsigned __int8 *)v24;
    v27 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v25 + 16LL);
    if ( v27 == sub_9202E0 )
    {
      if ( *v9 > 0x15u || *v26 > 0x15u )
        goto LABEL_29;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v28 = sub_AD5570(28, (__int64)v9, v26, 0, 0);
      else
        v28 = sub_AABE40(0x1Cu, v9, v26);
    }
    else
    {
      v28 = v27(v25, 28u, v9, v26);
    }
    if ( v28 )
    {
LABEL_24:
      v14 = *(_QWORD **)(a1 + 920);
      v9 = (unsigned __int8 *)v28;
      goto LABEL_9;
    }
LABEL_29:
    v46 = 257;
    v41 = sub_B504D0(28, (__int64)v9, (__int64)v26, (__int64)v45, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v41,
      v43,
      a3[7],
      a3[8]);
    v33 = *a3;
    v28 = v41;
    v34 = *a3 + 16LL * *((unsigned int *)a3 + 2);
    if ( *a3 != v34 )
    {
      do
      {
        v35 = *(_QWORD *)(v33 + 8);
        v36 = *(_DWORD *)v33;
        v33 += 16;
        v42 = v28;
        sub_B99FD0(v28, v36, v35);
        v28 = v42;
      }
      while ( v34 != v33 );
    }
    goto LABEL_24;
  }
LABEL_9:
  v15 = v14[1];
  if ( v15 )
  {
    v17 = *(_QWORD *)(a1 + 64);
    v44 = 257;
    v18 = sub_ACD640(v17, v15, 0);
    v19 = a3[10];
    v20 = (unsigned __int8 *)v18;
    v21 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v19 + 16LL);
    if ( v21 == sub_9202E0 )
    {
      if ( *v9 > 0x15u || *v20 > 0x15u )
        goto LABEL_32;
      if ( (unsigned __int8)sub_AC47B0(30) )
        v22 = sub_AD5570(30, (__int64)v9, v20, 0, 0);
      else
        v22 = sub_AABE40(0x1Eu, v9, v20);
    }
    else
    {
      v22 = v21(v19, 30u, v9, v20);
    }
    if ( v22 )
      return (unsigned __int8 *)v22;
LABEL_32:
    v46 = 257;
    v22 = sub_B504D0(30, (__int64)v9, (__int64)v20, (__int64)v45, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v22,
      v43,
      a3[7],
      a3[8]);
    v37 = *a3;
    v38 = *a3 + 16LL * *((unsigned int *)a3 + 2);
    while ( v38 != v37 )
    {
      v39 = *(_QWORD *)(v37 + 8);
      v40 = *(_DWORD *)v37;
      v37 += 16;
      sub_B99FD0(v22, v40, v39);
    }
    return (unsigned __int8 *)v22;
  }
  return v9;
}
