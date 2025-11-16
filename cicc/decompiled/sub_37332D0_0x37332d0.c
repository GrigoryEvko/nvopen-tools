// Function: sub_37332D0
// Address: 0x37332d0
//
void __fastcall sub_37332D0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 *v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r12
  _QWORD *v12; // rdx
  void (__fastcall **v13[8])(_QWORD, _QWORD, _QWORD *); // [rsp+0h] [rbp-40h] BYREF

  v2 = *a2;
  v13[1] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD *))a1;
  v3 = 32 * v2;
  v13[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD *))&unk_4A3D398;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 760LL);
  v5 = *(_QWORD *)(v4 + 1288);
  v6 = (__int64 *)(v5 + v3);
  v7 = *(_QWORD *)(v5 + v3 + 16);
  if ( (v3 >> 5) + 1 == *(_DWORD *)(v4 + 1296) )
    v8 = *(unsigned int *)(v4 + 1440);
  else
    v8 = *(_QWORD *)(v5 + v3 + 48);
  v9 = v8 - v7;
  v10 = *(_QWORD *)(v4 + 1432) + 32 * v7;
  v11 = v10 + 32 * v9;
  while ( v11 != v10 )
  {
    v12 = (_QWORD *)v10;
    v10 += 32;
    sub_321FA30(v4, v13, v12, *v6);
  }
}
