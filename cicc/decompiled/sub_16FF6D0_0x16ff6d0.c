// Function: sub_16FF6D0
// Address: 0x16ff6d0
//
__int64 __fastcall sub_16FF6D0(unsigned int a1)
{
  __int64 v1; // rbp
  char *v3; // rax
  unsigned int v4; // r9d
  unsigned int v5; // r10d
  unsigned int v6; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( a1 > 0x10FFFF )
    return 0;
  v7 = v1;
  v6 = a1;
  v3 = (char *)sub_16FF680((__int64)&aNoTrappingMath[-4384], (__int64)"no-trapping-math", &v6);
  if ( v3 == "no-trapping-math" || v4 < *(_DWORD *)v3 )
    return 1;
  else
    return v5;
}
