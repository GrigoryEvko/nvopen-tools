// Function: sub_1688540
// Address: 0x1688540
//
size_t __fastcall sub_1688540(__int64 *a1, const char *a2, __m128i *a3)
{
  int v5; // eax
  __int64 v6; // rdx
  size_t v7; // r15
  __int64 v9; // rdi
  __int64 v10; // rsi
  char *v11; // rax
  int v12; // edx
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  char *v16; // rbx
  char v17; // [rsp+0h] [rbp-460h]
  __m128i arg; // [rsp+18h] [rbp-448h] BYREF
  __int64 v20; // [rsp+28h] [rbp-438h]
  char s[1072]; // [rsp+30h] [rbp-430h] BYREF

  arg = _mm_loadu_si128(a3);
  v20 = a3[1].m128i_i64[0];
  v5 = vsnprintf(s, 0x400u, a2, &arg);
  v7 = v5;
  if ( (unsigned __int64)v5 > 0x3FF )
  {
    v9 = *(_QWORD *)(sub_1689050(s, 1024, v6) + 24);
    v10 = (int)v7 + 1;
    v11 = (char *)sub_1685080(v9, v10);
    v16 = v11;
    if ( v11 )
    {
      v7 = vsprintf(v11, a2, a3);
      sub_1688330(a1, v16, v7);
      if ( v16 != s )
        sub_16856A0(v16);
    }
    else
    {
      sub_1683C30(v9, v10, v12, v13, v14, v15, v17);
      return 0;
    }
  }
  else
  {
    sub_1688330(a1, s, v5);
  }
  return v7;
}
