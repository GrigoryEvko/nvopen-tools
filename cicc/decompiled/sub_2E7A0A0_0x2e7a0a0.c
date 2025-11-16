// Function: sub_2E7A0A0
// Address: 0x2e7a0a0
//
__int64 __fastcall sub_2E7A0A0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r8
  __int64 (__fastcall *v3)(__int64, __int64); // rax
  __int64 v4; // r13
  char v5; // bl
  __int64 v6; // rax
  char v7; // cl
  __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v12; // r13
  char v13; // bl
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  __int64 v17; // [rsp+8h] [rbp-28h]

  v2 = *(_QWORD **)a1;
  if ( !*(_BYTE *)(a1 + 9) )
  {
    v12 = v2[1];
    v13 = sub_AE5020(a2, v12);
    v14 = sub_9208B0(a2, v12);
    v7 = v13;
    v17 = v15;
    v10 = (unsigned __int64)(v14 + 7) >> 3;
    v9 = 1LL << v13;
    goto LABEL_4;
  }
  v3 = *(__int64 (__fastcall **)(__int64, __int64))(*v2 + 24LL);
  if ( v3 == sub_2E788D0 )
  {
    v4 = v2[1];
    v5 = sub_AE5020(a2, v4);
    v6 = sub_9208B0(a2, v4);
    v7 = v5;
    v17 = v8;
    v9 = 1LL << v5;
    v10 = (unsigned __int64)(v6 + 7) >> 3;
LABEL_4:
    v16 = (v10 + v9 - 1) >> v7 << v7;
    return sub_CA1930(&v16);
  }
  return ((__int64 (__fastcall *)(_QWORD))v3)(*(_QWORD *)a1);
}
