// Function: sub_3906D80
// Address: 0x3906d80
//
__int64 __fastcall sub_3906D80(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  char v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // r12d
  size_t v9; // rdx
  _BYTE *v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  const char *v15; // rax
  __int64 v16; // rdi
  _QWORD v18[2]; // [rsp+0h] [rbp-60h] BYREF
  void *s; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  _QWORD v21[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v22; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v18[0] = 0;
  v18[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v18) )
    goto LABEL_13;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    HIBYTE(v22) = 1;
    v15 = "expected a comma";
LABEL_11:
    v16 = *(_QWORD *)(a1 + 8);
    v21[0] = v15;
    LOBYTE(v22) = 3;
    return (unsigned int)sub_3909CF0(v16, v21, 0, 0, v3, v4);
  }
  v5 = *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113);
  *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113) = 1;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  s = 0;
  *(_BYTE *)(v6 + 113) = v5;
  v7 = *(_QWORD *)(a1 + 8);
  v20 = 0;
  v8 = (*(__int64 (__fastcall **)(__int64, void **))(*(_QWORD *)v7 + 144LL))(v7, &s);
  if ( (_BYTE)v8 )
  {
LABEL_13:
    HIBYTE(v22) = 1;
    v15 = "expected identifier in directive";
    goto LABEL_11;
  }
  v9 = v20;
  if ( !v20 )
    goto LABEL_14;
  v10 = s;
  if ( v20 < 0 )
    v9 = 0x7FFFFFFFFFFFFFFFLL;
  v11 = memchr(s, 64, v9);
  if ( !v11 || v11 - v10 == -1 )
  {
LABEL_14:
    HIBYTE(v22) = 1;
    v15 = "expected a '@' in the name";
    goto LABEL_11;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v21[0] = v18;
  v22 = 261;
  v13 = sub_38BF510(v12, (__int64)v21);
  v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, void *, __int64, __int64))(*(_QWORD *)v14 + 352LL))(v14, s, v20, v13);
  return v8;
}
