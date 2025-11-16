// Function: sub_1B99770
// Address: 0x1b99770
//
bool __fastcall sub_1B99770(__int64 *a1, int *a2)
{
  __int64 v2; // rbp
  int v3; // edx
  bool result; // al
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // [rsp-30h] [rbp-30h] BYREF
  __int64 v8; // [rsp-28h] [rbp-28h] BYREF
  int v9; // [rsp-20h] [rbp-20h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  v3 = *a2;
  result = 0;
  if ( (unsigned int)*a2 > 1 )
  {
    v10 = v2;
    v5 = *a1;
    v9 = v3;
    v6 = *(_QWORD *)(v5 + 32);
    v8 = a1[1];
    return (unsigned __int8)sub_1B99450(v6 + 264, &v8, &v7)
        && v7 != *(_QWORD *)(v6 + 272) + 24LL * *(unsigned int *)(v6 + 288)
        && *(_DWORD *)(v7 + 16) == 3;
  }
  return result;
}
