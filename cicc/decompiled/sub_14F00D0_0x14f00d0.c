// Function: sub_14F00D0
// Address: 0x14f00d0
//
__int64 __fastcall sub_14F00D0(__int64 a1, unsigned int *a2)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 *v5; // r12
  __int64 result; // rax
  _BYTE *v7; // rsi
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_QWORD **)a1;
  v3 = *a2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 528LL);
  if ( v3 >= (*(_QWORD *)(*(_QWORD *)a1 + 536LL) - v4) >> 3 )
    return 0;
  v5 = (__int64 *)(v4 + 8 * v3);
  result = *v5;
  if ( !*v5 )
  {
    result = sub_16440F0(v2[54]);
    v7 = (_BYTE *)v2[224];
    v8[0] = result;
    if ( v7 == (_BYTE *)v2[225] )
    {
      sub_14EFD20((__int64)(v2 + 223), v7, v8);
      result = v8[0];
    }
    else
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = result;
        v7 = (_BYTE *)v2[224];
      }
      v2[224] = v7 + 8;
    }
    *v5 = result;
  }
  return result;
}
