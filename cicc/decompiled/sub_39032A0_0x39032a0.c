// Function: sub_39032A0
// Address: 0x39032a0
//
__int64 __fastcall sub_39032A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r13
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v19; // rdi
  const char *v20; // rax
  _QWORD v21[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v22[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v7 = *(unsigned int *)(v6 + 120);
  if ( !(_DWORD)v7 )
    BUG();
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v6 + 112) + 32 * v7 - 32) + 184LL);
  if ( (unsigned int)(v9 - 6) > 2 && v9 != 20 )
  {
    v22[0] = "indirect symbol not in a symbol pointer or stub section";
    v23 = 259;
    return (unsigned int)sub_3909790(v8, a4, v22, 0, 0);
  }
  v21[0] = 0;
  v21[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v8 + 144LL))(v8, v21) )
  {
    v19 = *(_QWORD *)(a1 + 8);
    v22[0] = "expected identifier in .indirect_symbol directive";
    v23 = 259;
    return (unsigned int)sub_3909CF0(v19, v22, 0, 0, v10, v11);
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v22[0] = v21;
  v23 = 261;
  v15 = sub_38BF510(v12, (__int64)v22);
  v16 = *(_BYTE *)(v15 + 8) & 1;
  if ( (*(_BYTE *)(v15 + 8) & 1) != 0 )
  {
    HIBYTE(v23) = 1;
    v20 = "non-local symbol required in directive";
LABEL_13:
    v22[0] = v20;
    LOBYTE(v23) = 3;
    return (unsigned int)sub_3909CF0(*(_QWORD *)(a1 + 8), v22, 0, 0, v13, v14);
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v17 + 256LL))(v17, v15, 10) )
  {
    v22[1] = v21;
    v23 = 1283;
    v22[0] = "unable to emit indirect symbol attribute for: ";
    return (unsigned int)sub_3909CF0(*(_QWORD *)(a1 + 8), v22, 0, 0, v13, v14);
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v23) = 1;
    v20 = "unexpected token in '.indirect_symbol' directive";
    goto LABEL_13;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  return v16;
}
