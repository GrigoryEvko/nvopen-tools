// Function: sub_393B9A0
// Address: 0x393b9a0
//
__int64 *__fastcall sub_393B9A0(__int64 *a1, _QWORD *a2, unsigned __int64 *a3)
{
  bool (__fastcall *v4)(__int64); // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *(bool (__fastcall **)(__int64))(*a2 + 40LL);
  if ( v4 == sub_3937D40 )
  {
    if ( !a2[4] )
    {
LABEL_3:
      v8[0] = 1;
      sub_3939440(a1, v8);
      return a1;
    }
  }
  else if ( v4((__int64)a2) )
  {
    goto LABEL_3;
  }
  v6 = a2[2] + (a2[3] == 0 ? 10LL : 8LL);
  *a3 = sub_393B300(a2[5], v6 + 16, *(_QWORD *)v6, (unsigned int *)(v6 + 16 + *(_QWORD *)v6), *(_QWORD *)(v6 + 8));
  a3[1] = v7;
  if ( v7 )
  {
    *a1 = 1;
  }
  else
  {
    v8[0] = 9;
    sub_3939440(a1, v8);
  }
  return a1;
}
