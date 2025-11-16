// Function: sub_B1DED0
// Address: 0xb1ded0
//
bool __fastcall sub_B1DED0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // r13
  __int64 v10; // rax
  int v11; // eax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v14; // [rsp+8h] [rbp-28h]

  v5 = a2[1];
  v6 = *(_QWORD *)a1;
  v13 = *a2;
  v14 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = sub_B1DDD0(v6, &v13);
  v8 = *(_QWORD *)a1;
  v9 = v7;
  v10 = a3[1];
  v13 = *a3;
  v14 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *(_DWORD *)sub_B1DDD0(v8, &v13);
  if ( **(_BYTE **)(a1 + 8) )
    return *(_DWORD *)v9 < v11;
  else
    return *(_DWORD *)v9 > v11;
}
