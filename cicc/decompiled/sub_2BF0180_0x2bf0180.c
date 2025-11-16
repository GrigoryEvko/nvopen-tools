// Function: sub_2BF0180
// Address: 0x2bf0180
//
__int64 __fastcall sub_2BF0180(unsigned int *a1, __int64 a2, __int64 *a3)
{
  char v3; // al
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int8 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int8 *v14; // r14
  __int64 (__fastcall *v15)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  char v21[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v22; // [rsp+20h] [rbp-70h]
  char v23[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v24; // [rsp+50h] [rbp-40h]

  v3 = *((_BYTE *)a1 + 4);
  if ( !v3 )
  {
    v4 = *a1;
    v5 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
    return sub_ACD640(v5, v4, 0);
  }
  if ( v3 != 1 )
    BUG();
  v8 = *(_DWORD *)a3 - *a1;
  v22 = 257;
  v9 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
  v10 = (unsigned __int8 *)sub_ACD640(v9, v8, 0);
  v11 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
  v12 = sub_2AB2710(a2, v11, *a3);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = (unsigned __int8 *)v12;
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v13 + 32LL);
  if ( v15 == sub_9201A0 )
  {
    if ( *v14 > 0x15u || *v10 > 0x15u )
    {
LABEL_11:
      v24 = 257;
      v16 = sub_B504D0(15, (__int64)v14, (__int64)v10, (__int64)v23, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v16,
        v21,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v17 = *(_QWORD *)a2;
      v18 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v18 )
      {
        do
        {
          v19 = *(_QWORD *)(v17 + 8);
          v20 = *(_DWORD *)v17;
          v17 += 16;
          sub_B99FD0(v16, v20, v19);
        }
        while ( v18 != v17 );
      }
      return v16;
    }
    if ( (unsigned __int8)sub_AC47B0(15) )
      v16 = sub_AD5570(15, (__int64)v14, v10, 0, 0);
    else
      v16 = sub_AABE40(0xFu, v14, v10);
  }
  else
  {
    v16 = v15(v13, 15u, v14, v10, 0, 0);
  }
  if ( !v16 )
    goto LABEL_11;
  return v16;
}
