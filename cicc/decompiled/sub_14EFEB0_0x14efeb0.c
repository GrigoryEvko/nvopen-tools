// Function: sub_14EFEB0
// Address: 0x14efeb0
//
__int64 __fastcall sub_14EFEB0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 *v3; // r12
  __int64 result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp-20h] [rbp-20h] BYREF

  v2 = a1[66];
  if ( a2 >= (unsigned __int64)((a1[67] - v2) >> 3) )
    return 0;
  v3 = (__int64 *)(v2 + 8LL * a2);
  result = *v3;
  if ( !*v3 )
  {
    result = sub_16440F0(a1[54]);
    v5 = (_BYTE *)a1[224];
    v6 = result;
    if ( v5 == (_BYTE *)a1[225] )
    {
      sub_14EFD20((__int64)(a1 + 223), v5, &v6);
      result = v6;
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = result;
        v5 = (_BYTE *)a1[224];
      }
      a1[224] = v5 + 8;
    }
    *v3 = result;
  }
  return result;
}
