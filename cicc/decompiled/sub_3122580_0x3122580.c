// Function: sub_3122580
// Address: 0x3122580
//
__int64 __fastcall sub_3122580(__int64 *a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v10; // r12
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  _BYTE v21[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v8 = a1[10];
  v9 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v8 + 24LL);
  if ( v9 != sub_920250 )
  {
    v10 = v9(v8, 19u, a2, a3, a5);
    goto LABEL_6;
  }
  if ( *a2 <= 0x15u && *a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(19) )
      v10 = sub_AD5570(19, (__int64)a2, a3, a5, 0);
    else
      v10 = sub_AABE40(0x13u, a2, a3);
LABEL_6:
    if ( v10 )
      return v10;
  }
  if ( a5 )
  {
    v22 = 257;
    v10 = sub_B504D0(19, (__int64)a2, (__int64)a3, (__int64)v21, 0, 0);
    sub_B448B0(v10, 1);
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a4,
      a1[7],
      a1[8]);
    v16 = *a1;
    v17 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    while ( v17 != v16 )
    {
      v18 = *(_QWORD *)(v16 + 8);
      v19 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v10, v19, v18);
    }
  }
  else
  {
    v22 = 257;
    v10 = sub_B504D0(19, (__int64)a2, (__int64)a3, (__int64)v21, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a4,
      a1[7],
      a1[8]);
    v12 = *a1;
    v13 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    while ( v13 != v12 )
    {
      v14 = *(_QWORD *)(v12 + 8);
      v15 = *(_DWORD *)v12;
      v12 += 16;
      sub_B99FD0(v10, v15, v14);
    }
  }
  return v10;
}
