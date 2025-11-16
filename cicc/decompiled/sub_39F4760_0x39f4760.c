// Function: sub_39F4760
// Address: 0x39f4760
//
__int64 __fastcall sub_39F4760(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, char a5)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v12; // [rsp+18h] [rbp-48h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *a2;
  *a2 = 0;
  v6 = *a3;
  *a3 = 0;
  v7 = *a4;
  *a4 = 0;
  v8 = sub_22077B0(0x148u);
  v9 = v8;
  if ( v8 )
  {
    v14[0] = v7;
    v13 = v6;
    v12 = v5;
    sub_38D3FD0(v8, a1, &v12, &v13, v14);
    if ( v12 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    if ( v14[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
    *(_BYTE *)(v9 + 320) = 0;
    *(_QWORD *)v9 = &unk_4A41828;
  }
  else
  {
    if ( v7 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    if ( v5 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  }
  if ( a5 )
    *(_BYTE *)(*(_QWORD *)(v9 + 264) + 484LL) |= 1u;
  return v9;
}
