// Function: sub_319A760
// Address: 0x319a760
//
__int64 __fastcall sub_319A760(__int64 *a1, __int64 a2, unsigned __int8 *a3, int a4)
{
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v10; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int8 *v14; // r15
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // esi
  _BYTE v24[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v25; // [rsp+20h] [rbp-70h]
  _BYTE v26[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v27; // [rsp+50h] [rbp-40h]

  v25 = 257;
  if ( !a4 || (v7 = (unsigned int)(a4 - 1), ((unsigned int)v7 & a4) != 0) )
  {
    v8 = a1[10];
    v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v8 + 16LL);
    if ( v9 == sub_9202E0 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *a3 > 0x15u )
      {
LABEL_19:
        v27 = 257;
        v10 = sub_B504D0(22, a2, (__int64)a3, (__int64)v26, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
          a1[11],
          v10,
          v24,
          a1[7],
          a1[8]);
        v20 = *a1;
        v21 = *a1 + 16LL * *((unsigned int *)a1 + 2);
        while ( v21 != v20 )
        {
          v22 = *(_QWORD *)(v20 + 8);
          v23 = *(_DWORD *)v20;
          v20 += 16;
          sub_B99FD0(v10, v23, v22);
        }
        return v10;
      }
      if ( (unsigned __int8)sub_AC47B0(22) )
        v10 = sub_AD5570(22, a2, a3, 0, 0);
      else
        v10 = sub_AABE40(0x16u, (unsigned __int8 *)a2, a3);
    }
    else
    {
      v10 = v9(v8, 22u, (_BYTE *)a2, a3);
    }
    if ( v10 )
      return v10;
    goto LABEL_19;
  }
  v12 = sub_AD64C0(*(_QWORD *)(a2 + 8), v7, 0);
  v13 = a1[10];
  v14 = (unsigned __int8 *)v12;
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v13 + 16LL);
  if ( v15 != sub_9202E0 )
  {
    v10 = v15(v13, 28u, (_BYTE *)a2, v14);
    goto LABEL_15;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v14 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v10 = sub_AD5570(28, a2, v14, 0, 0);
    else
      v10 = sub_AABE40(0x1Cu, (unsigned __int8 *)a2, v14);
LABEL_15:
    if ( v10 )
      return v10;
  }
  v27 = 257;
  v10 = sub_B504D0(28, a2, (__int64)v14, (__int64)v26, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v10,
    v24,
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
  return v10;
}
