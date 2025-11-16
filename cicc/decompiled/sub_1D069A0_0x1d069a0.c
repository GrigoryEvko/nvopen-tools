// Function: sub_1D069A0
// Address: 0x1d069a0
//
__int64 __fastcall sub_1D069A0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // r13
  char *v3; // r8
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 result; // rax
  unsigned __int64 v7; // rax
  char *v8; // rax
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1 + 12;
  v3 = (char *)a1[13];
  v4 = a1[12];
  v5 = (__int64)&v3[-v4] >> 2;
  if ( (unsigned int)v5 < 0xF0F0F0F0F0F0F0F1LL * ((__int64)(*(_QWORD *)(a1[6] + 8LL) - *(_QWORD *)a1[6]) >> 4) )
  {
    v7 = (unsigned int)(2 * v5);
    v9[0] = 0;
    if ( v7 > v5 )
    {
      sub_1CFD340((__int64)v2, v3, v7 - v5, v9);
      v4 = a1[12];
    }
    else if ( v7 < v5 )
    {
      v8 = (char *)(v4 + 4 * v7);
      if ( v3 != v8 )
        a1[13] = v8;
    }
  }
  result = *(unsigned int *)(v4 + 4LL * *(unsigned int *)(a2 + 192));
  if ( !(_DWORD)result )
    return sub_1D01FD0(a2, v2);
  return result;
}
