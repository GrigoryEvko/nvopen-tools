// Function: sub_3356A00
// Address: 0x3356a00
//
__int64 __fastcall sub_3356A00(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // r13
  char *v4; // r8
  __int64 v5; // rdi
  unsigned __int64 v6; // rdx
  __int64 result; // rax
  unsigned __int64 v8; // rax
  char *v9; // rax
  _DWORD v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = a1 + 12;
  v4 = (char *)a1[13];
  v5 = a1[12];
  v6 = (__int64)&v4[-v5] >> 2;
  if ( (unsigned int)v6 < (unsigned __int64)((__int64)(*(_QWORD *)(a1[6] + 8LL) - *(_QWORD *)a1[6]) >> 8) )
  {
    v8 = (unsigned int)(2 * v6);
    v10[0] = 0;
    if ( v8 > v6 )
    {
      sub_1CFD340((__int64)v3, v4, v8 - v6, v10);
      v5 = a1[12];
    }
    else if ( v8 < v6 )
    {
      v9 = (char *)(v5 + 4 * v8);
      if ( v4 != v9 )
        a1[13] = v9;
    }
  }
  result = *(unsigned int *)(v5 + 4LL * *(unsigned int *)(a2 + 200));
  if ( !(_DWORD)result )
    return sub_3352560(a2, v3);
  return result;
}
