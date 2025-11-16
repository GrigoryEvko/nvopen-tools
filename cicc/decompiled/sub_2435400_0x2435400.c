// Function: sub_2435400
// Address: 0x2435400
//
__int64 __fastcall sub_2435400(char a1, char a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int8 *v12; // r15
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v14; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int8 *v18; // r15
  __int64 (__fastcall *v19)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  _BYTE v28[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v29; // [rsp+20h] [rbp-70h]
  _BYTE v30[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v31; // [rsp+50h] [rbp-40h]

  v8 = a3 << a2;
  v29 = 257;
  v9 = *(_QWORD *)(a5 + 8);
  if ( a1 )
  {
    v10 = sub_AD64C0(v9, v8, 0);
    v11 = a4[10];
    v12 = (unsigned __int8 *)v10;
    v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v11 + 16LL);
    if ( v13 == sub_9202E0 )
    {
      if ( *(_BYTE *)a5 > 0x15u || *v12 > 0x15u )
      {
LABEL_18:
        v31 = 257;
        v14 = sub_B504D0(29, a5, (__int64)v12, (__int64)v30, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
          a4[11],
          v14,
          v28,
          a4[7],
          a4[8]);
        v24 = *a4;
        v25 = *a4 + 16LL * *((unsigned int *)a4 + 2);
        while ( v25 != v24 )
        {
          v26 = *(_QWORD *)(v24 + 8);
          v27 = *(_DWORD *)v24;
          v24 += 16;
          sub_B99FD0(v14, v27, v26);
        }
        return v14;
      }
      if ( (unsigned __int8)sub_AC47B0(29) )
        v14 = sub_AD5570(29, a5, v12, 0, 0);
      else
        v14 = sub_AABE40(0x1Du, (unsigned __int8 *)a5, v12);
    }
    else
    {
      v14 = v13(v11, 29u, (_BYTE *)a5, v12);
    }
    if ( v14 )
      return v14;
    goto LABEL_18;
  }
  v16 = sub_AD64C0(v9, ~v8, 0);
  v17 = a4[10];
  v18 = (unsigned __int8 *)v16;
  v19 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v17 + 16LL);
  if ( v19 != sub_9202E0 )
  {
    v14 = v19(v17, 28u, (_BYTE *)a5, v18);
    goto LABEL_14;
  }
  if ( *(_BYTE *)a5 <= 0x15u && *v18 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v14 = sub_AD5570(28, a5, v18, 0, 0);
    else
      v14 = sub_AABE40(0x1Cu, (unsigned __int8 *)a5, v18);
LABEL_14:
    if ( v14 )
      return v14;
  }
  v31 = 257;
  v14 = sub_B504D0(28, a5, (__int64)v18, (__int64)v30, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
    a4[11],
    v14,
    v28,
    a4[7],
    a4[8]);
  v20 = *a4;
  v21 = *a4 + 16LL * *((unsigned int *)a4 + 2);
  while ( v21 != v20 )
  {
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_DWORD *)v20;
    v20 += 16;
    sub_B99FD0(v14, v23, v22);
  }
  return v14;
}
