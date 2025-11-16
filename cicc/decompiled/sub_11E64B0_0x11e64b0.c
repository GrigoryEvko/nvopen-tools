// Function: sub_11E64B0
// Address: 0x11e64b0
//
__int64 __fastcall sub_11E64B0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int8 *v7; // r14
  unsigned __int8 *v8; // r15
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v10; // r13
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdx
  unsigned int v15; // esi
  char v16[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v17; // [rsp+20h] [rbp-70h]
  char v18[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  v17 = 257;
  v5 = sub_AD64C0(v4, 127, 0);
  v6 = a3[10];
  v7 = (unsigned __int8 *)v5;
  v8 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v6 + 16LL);
  if ( v9 != sub_9202E0 )
  {
    v10 = v9(v6, 28u, *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v7);
    goto LABEL_6;
  }
  if ( *v8 <= 0x15u && *v7 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v10 = sub_AD5570(28, (__int64)v8, v7, 0, 0);
    else
      v10 = sub_AABE40(0x1Cu, v8, v7);
LABEL_6:
    if ( v10 )
      return v10;
  }
  v19 = 257;
  v10 = sub_B504D0(28, (__int64)v8, (__int64)v7, (__int64)v18, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v10,
    v16,
    a3[7],
    a3[8]);
  v12 = *a3;
  v13 = *a3 + 16LL * *((unsigned int *)a3 + 2);
  while ( v13 != v12 )
  {
    v14 = *(_QWORD *)(v12 + 8);
    v15 = *(_DWORD *)v12;
    v12 += 16;
    sub_B99FD0(v10, v15, v14);
  }
  return v10;
}
