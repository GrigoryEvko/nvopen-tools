// Function: sub_FF1A00
// Address: 0xff1a00
//
__int64 __fastcall sub_FF1A00(__int64 a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int *v4; // r14
  __int64 v5; // r13
  unsigned int *v6; // rbx
  __int64 v7; // rdx
  _BOOL4 v8; // r15d
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  result = 9 * a3;
  v4 = &a2[18 * a3];
  v5 = a1 + 8;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 8;
  *(_QWORD *)(a1 + 32) = a1 + 8;
  *(_QWORD *)(a1 + 40) = 0;
  if ( v4 != a2 )
  {
    v6 = a2;
    do
    {
      result = sub_FF1900((_QWORD *)a1, v5, v6);
      if ( v7 )
      {
        v8 = result || v5 == v7 || *v6 < *(_DWORD *)(v7 + 32);
        v15 = v7;
        v9 = sub_22077B0(104);
        v12 = v15;
        v13 = v9;
        *(_DWORD *)(v9 + 32) = *v6;
        *(_QWORD *)(v9 + 40) = v9 + 56;
        *(_QWORD *)(v9 + 48) = 0xC00000000LL;
        if ( v6[4] )
        {
          v14 = v15;
          v16 = v9;
          sub_FEE1E0(v9 + 40, (__int64)(v6 + 2), v12, v10, v9, v11);
          v12 = v14;
          v13 = v16;
        }
        result = sub_220F040(v8, v13, v12, v5);
        ++*(_QWORD *)(a1 + 40);
      }
      v6 += 18;
    }
    while ( v4 != v6 );
  }
  return result;
}
