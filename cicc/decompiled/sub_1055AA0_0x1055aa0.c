// Function: sub_1055AA0
// Address: 0x1055aa0
//
unsigned __int64 __fastcall sub_1055AA0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  unsigned __int64 result; // rax
  __int64 *v5; // rax
  unsigned __int64 v6; // [rsp-90h] [rbp-90h]
  __int64 v7; // [rsp-80h] [rbp-80h] BYREF
  _QWORD v8[2]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v9; // [rsp-68h] [rbp-68h]
  __int64 v10; // [rsp-60h] [rbp-60h] BYREF
  unsigned int v11; // [rsp-58h] [rbp-58h]
  int v12; // [rsp-20h] [rbp-20h] BYREF
  __int64 (__fastcall *v13)(__int64 *, __int64); // [rsp-18h] [rbp-18h]
  __int64 *v14; // [rsp-10h] [rbp-10h]
  __int64 v15; // [rsp-8h] [rbp-8h]

  result = a1;
  if ( *(_DWORD *)(a2 + 20) != *(_DWORD *)(a2 + 24) )
  {
    v15 = v3;
    v7 = a2;
    v5 = &v10;
    v8[1] = 0;
    v9 = 1;
    v8[0] = a3;
    do
    {
      *v5 = -4096;
      v5 += 2;
    }
    while ( v5 != (__int64 *)&v12 );
    v14 = &v7;
    v12 = 1;
    v13 = sub_1054BB0;
    result = sub_1054C20((__int64)v8, a1);
    if ( (v9 & 1) == 0 )
    {
      v6 = result;
      sub_C7D6A0(v10, 16LL * v11, 8);
      return v6;
    }
  }
  return result;
}
