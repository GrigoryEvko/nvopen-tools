// Function: sub_2623BF0
// Address: 0x2623bf0
//
__int64 __fastcall sub_2623BF0(_QWORD *a1, void *a2, void *a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // r13
  __int64 **v14; // rax
  __int64 *v16; // rsi
  unsigned __int8 *v17; // r15
  __int64 v18; // rbx
  __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29[8]; // [rsp+0h] [rbp-40h] BYREF

  v9 = *a1;
  if ( (unsigned int)(*(_DWORD *)(*a1 + 28LL) - 38) <= 1 && *(_DWORD *)(v9 + 36) == 3 )
  {
    v14 = (__int64 **)a1[1];
    v16 = v14[1];
    v12 = sub_2623AF0(*v14, v16, a2, a3);
    v17 = sub_BD3990(v12, (__int64)v16);
    if ( *(_BYTE *)(a6 + 8) == 12 )
      v12 = (unsigned __int8 *)sub_AD4C50((unsigned __int64)v12, (__int64 **)a6, 0);
    if ( (v17[7] & 0x20) == 0 || !sub_B91C10((__int64)v17, 21) )
    {
      v18 = *a1;
      v19 = *(_QWORD *)(*a1 + 112LL);
      if ( a5 == *(_DWORD *)(v19 + 8) >> 8 )
      {
        v27 = sub_ACD640(v19, -1, 0);
        v28 = sub_B98A20(v27, -1);
        v22 = *(_QWORD *)(v18 + 112);
        v24 = -1;
        v23 = (__int64)v28;
      }
      else
      {
        v20 = sub_ACD640(v19, 0, 0);
        v21 = sub_B98A20(v20, 0);
        v22 = *(_QWORD *)(v18 + 112);
        v23 = (__int64)v21;
        v24 = 1LL << a5;
      }
      v25 = sub_ACD640(v22, v24, 0);
      v29[0] = v23;
      v29[1] = (__int64)sub_B98A20(v25, v24);
      v26 = sub_B9C770(**(__int64 ***)v18, v29, (__int64 *)2, 0, 1);
      sub_B99110((__int64)v17, 21, v26);
    }
    return (__int64)v12;
  }
  v10 = a6;
  if ( *(_BYTE *)(a6 + 8) != 12 )
    v10 = *(_QWORD *)(v9 + 104);
  v11 = sub_AD64C0(v10, a4, 0);
  v12 = (unsigned __int8 *)v11;
  if ( *(_BYTE *)(a6 + 8) == 12 )
    return (__int64)v12;
  return sub_AD4C70(v11, (__int64 **)a6, 0);
}
