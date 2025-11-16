// Function: sub_1E0CED0
// Address: 0x1e0ced0
//
__int64 __fastcall sub_1E0CED0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  int v6; // r13d
  __int64 result; // rax
  __int64 v9; // r14
  __int64 *i; // r15
  _BYTE *v11; // rsi
  int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-48h]
  _DWORD v14[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = a4;
  result = sub_1E0C9D0(a1, a2, a3, a4, a5, a6);
  if ( v6 )
  {
    v9 = result;
    v13 = result + 96;
    for ( i = (__int64 *)(a3 + 8LL * (unsigned int)(v6 - 1)); ; i = (__int64 *)result )
    {
      v12 = sub_1E0C1F0(a1, *i);
      v11 = *(_BYTE **)(v9 + 104);
      v14[0] = v12;
      if ( v11 == *(_BYTE **)(v9 + 112) )
      {
        sub_1E0CD40(v13, v11, v14);
        result = (__int64)(i - 1);
        if ( (__int64 *)a3 == i )
          return result;
      }
      else
      {
        if ( v11 )
        {
          *(_DWORD *)v11 = v12;
          v11 = *(_BYTE **)(v9 + 104);
        }
        result = (__int64)(i - 1);
        *(_QWORD *)(v9 + 104) = v11 + 4;
        if ( (__int64 *)a3 == i )
          return result;
      }
    }
  }
  return result;
}
