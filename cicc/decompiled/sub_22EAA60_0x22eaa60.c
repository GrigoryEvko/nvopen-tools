// Function: sub_22EAA60
// Address: 0x22eaa60
//
bool __fastcall sub_22EAA60(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned __int64 v2; // r12
  bool v3; // cc
  bool result; // al
  bool v5; // [rsp-49h] [rbp-49h]
  const void *v6; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v7; // [rsp-40h] [rbp-40h]
  const void *v8; // [rsp-38h] [rbp-38h] BYREF
  unsigned int v9; // [rsp-30h] [rbp-30h]

  if ( (unsigned __int8)(*(_BYTE *)a1 - 4) > 1u )
    return *(_BYTE *)a1 == 2;
  v7 = *(_DWORD *)(a1 + 16);
  if ( v7 > 0x40 )
    sub_C43780((__int64)&v6, (const void **)(a1 + 8));
  else
    v6 = *(const void **)(a1 + 8);
  sub_C46A40((__int64)&v6, 1);
  v1 = v7;
  v2 = (unsigned __int64)v6;
  v7 = 0;
  v3 = *(_DWORD *)(a1 + 32) <= 0x40u;
  v9 = v1;
  v8 = v6;
  if ( v3 )
    result = *(_QWORD *)(a1 + 24) == (_QWORD)v6;
  else
    result = sub_C43C50(a1 + 24, &v8);
  if ( v1 > 0x40 )
  {
    if ( v2 )
    {
      v5 = result;
      j_j___libc_free_0_0(v2);
      result = v5;
      if ( v7 > 0x40 )
      {
        if ( v6 )
        {
          j_j___libc_free_0_0((unsigned __int64)v6);
          result = v5;
        }
      }
    }
  }
  if ( !result )
    return *(_BYTE *)a1 == 2;
  return result;
}
