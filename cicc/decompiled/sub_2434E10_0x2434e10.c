// Function: sub_2434E10
// Address: 0x2434e10
//
__int64 __fastcall sub_2434E10(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int8 *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int8 *v8; // rbx
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v10; // r15
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v13; // r13
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rax
  _BYTE v24[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v25; // [rsp+20h] [rbp-70h]
  _BYTE v26[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v27; // [rsp+50h] [rbp-40h]

  v3 = sub_2A3A5A0(a1 + 24);
  v4 = *(_QWORD *)(a1 + 528);
  v5 = (unsigned __int8 *)v3;
  if ( !v4 )
  {
    v23 = sub_2A3A780(a2);
    *(_QWORD *)(a1 + 528) = v23;
    v4 = v23;
  }
  v25 = 257;
  v6 = sub_AD64C0(*(_QWORD *)(v4 + 8), 44, 0);
  v7 = a2[10];
  v8 = (unsigned __int8 *)v6;
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v7 + 32LL);
  if ( v9 != sub_9201A0 )
  {
    v10 = (unsigned __int8 *)v9(v7, 25u, (_BYTE *)v4, v8, 0, 0);
    goto LABEL_8;
  }
  if ( *(_BYTE *)v4 <= 0x15u && *v8 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(25) )
      v10 = (unsigned __int8 *)sub_AD5570(25, v4, v8, 0, 0);
    else
      v10 = (unsigned __int8 *)sub_AABE40(0x19u, (unsigned __int8 *)v4, v8);
LABEL_8:
    if ( v10 )
      goto LABEL_9;
  }
  v27 = 257;
  v10 = (unsigned __int8 *)sub_B504D0(25, v4, (__int64)v8, (__int64)v26, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v10,
    v24,
    a2[7],
    a2[8]);
  v15 = *a2;
  v16 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v16 )
  {
    do
    {
      v17 = *(_QWORD *)(v15 + 8);
      v18 = *(_DWORD *)v15;
      v15 += 16;
      sub_B99FD0((__int64)v10, v18, v17);
    }
    while ( v16 != v15 );
  }
LABEL_9:
  v11 = a2[10];
  v25 = 257;
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v11 + 16LL);
  if ( v12 != sub_9202E0 )
  {
    v13 = v12(v11, 29u, v5, v10);
    goto LABEL_14;
  }
  if ( *v5 <= 0x15u && *v10 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(29) )
      v13 = sub_AD5570(29, (__int64)v5, v10, 0, 0);
    else
      v13 = sub_AABE40(0x1Du, v5, v10);
LABEL_14:
    if ( v13 )
      return v13;
  }
  v27 = 257;
  v13 = sub_B504D0(29, (__int64)v5, (__int64)v10, (__int64)v26, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v13,
    v24,
    a2[7],
    a2[8]);
  v19 = *a2;
  v20 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  while ( v20 != v19 )
  {
    v21 = *(_QWORD *)(v19 + 8);
    v22 = *(_DWORD *)v19;
    v19 += 16;
    sub_B99FD0(v13, v22, v21);
  }
  return v13;
}
