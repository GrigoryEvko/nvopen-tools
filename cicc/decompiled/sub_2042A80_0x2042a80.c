// Function: sub_2042A80
// Address: 0x2042a80
//
__int64 __fastcall sub_2042A80(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rax
  char *v4; // r8
  __int64 v5; // rdi
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rbx
  char *v13; // rdx
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  a1[2] = a2;
  v3 = a2[1];
  v4 = (char *)a1[4];
  v5 = a1[3];
  v14[0] = 0;
  v6 = 0xF0F0F0F0F0F0F0F1LL * ((v3 - *a2) >> 4);
  v7 = (__int64)&v4[-v5] >> 2;
  if ( v6 > v7 )
  {
    sub_1CFD340((__int64)(a1 + 3), v4, v6 - v7, v14);
    a2 = (__int64 *)a1[2];
    v3 = a2[1];
  }
  else if ( v6 < v7 )
  {
    v13 = (char *)(v5 - 0x3C3C3C3C3C3C3C3CLL * ((v3 - *a2) >> 4));
    if ( v4 != v13 )
    {
      a1[4] = v13;
      v3 = a2[1];
    }
  }
  v8 = *a2;
  result = 0xF0F0F0F0F0F0F0F1LL * ((v3 - *a2) >> 4);
  if ( (_DWORD)result )
  {
    v10 = 0;
    v11 = 272LL * (unsigned int)(result - 1);
    while ( 1 )
    {
      v12 = v10 + v8;
      result = sub_2042970((__int64)a1, (__int64 *)v12);
      *(_DWORD *)(v12 + 196) = 0;
      if ( v11 == v10 )
        break;
      v10 += 272;
      v8 = *(_QWORD *)a1[2];
    }
  }
  return result;
}
