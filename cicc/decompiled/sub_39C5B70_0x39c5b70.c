// Function: sub_39C5B70
// Address: 0x39c5b70
//
void __fastcall sub_39C5B70(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdx
  void (__fastcall **v12[6])(_QWORD, _QWORD, _QWORD); // [rsp+0h] [rbp-30h] BYREF

  v2 = *a2;
  v12[1] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD))a1;
  v3 = 32 * v2;
  v12[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD))&unk_4A40660;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 504LL);
  v5 = *(_QWORD *)(v4 + 1192);
  v6 = *(_QWORD *)(v5 + v3 + 16);
  if ( (v3 >> 5) + 1 == *(_DWORD *)(v4 + 1200) )
    v7 = *(unsigned int *)(v4 + 1344);
  else
    v7 = *(_QWORD *)(v5 + v3 + 48);
  v8 = v7 - v6;
  v9 = *(_QWORD *)(v4 + 1336) + 32 * v6;
  v10 = v9 + 32 * v8;
  while ( v10 != v9 )
  {
    v11 = v9;
    v9 += 32;
    sub_398A890(v4, v12, v11);
  }
}
