// Function: sub_37B4DA0
// Address: 0x37b4da0
//
void __fastcall sub_37B4DA0(_QWORD *a1, __int64 **a2)
{
  char *v2; // r8
  __int64 v3; // rcx
  __int64 *v4; // r13
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  char *v7; // rax
  __int64 *i; // rbx
  __int64 *v9; // rsi
  int v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = (char *)a1[4];
  v3 = a1[3];
  a1[2] = a2;
  v4 = a2[1];
  v10[0] = 0;
  v5 = ((char *)v4 - (char *)*a2) >> 8;
  v6 = (__int64)&v2[-v3] >> 2;
  if ( v5 > v6 )
  {
    sub_1CFD340((__int64)(a1 + 3), v2, v5 - v6, v10);
    a2 = (__int64 **)a1[2];
    v4 = a2[1];
  }
  else if ( v5 < v6 )
  {
    v7 = (char *)(v3 + 4 * v5);
    if ( v2 != v7 )
    {
      a1[4] = v7;
      v4 = a2[1];
    }
  }
  for ( i = *a2; v4 != i; *((_DWORD *)i - 13) = 0 )
  {
    v9 = i;
    i += 32;
    sub_37B4BA0((__int64)a1, v9);
  }
}
