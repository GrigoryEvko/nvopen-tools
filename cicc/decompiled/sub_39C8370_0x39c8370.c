// Function: sub_39C8370
// Address: 0x39c8370
//
__int64 __fastcall sub_39C8370(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rdi
  void *v7; // rax
  size_t v8; // rdx
  unsigned __int16 v9; // ax
  int v10; // r8d
  __int64 v11; // rax
  __int64 result; // rax
  int v13[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)a2;
  v6 = *(_QWORD *)(*(_QWORD *)a2 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)a2 + 8LL)));
  if ( v6 )
  {
    v7 = (void *)sub_161E970(v6);
    if ( v8 )
      sub_39A3F30(a1, a3, 3, v7, v8);
    v5 = *(_QWORD *)a2;
  }
  v9 = sub_398C0A0(a1[25]);
  if ( v5 )
  {
    if ( v9 > 4u )
    {
      v10 = *(_DWORD *)(v5 + 28) >> 3;
      if ( v10 )
      {
        v13[0] = 65551;
        sub_39A3560((__int64)a1, (__int64 *)(a3 + 8), 136, (__int64)v13, v10 & 0x1FFFFFFF);
      }
    }
  }
  sub_39A3750((__int64)a1, a3, v5);
  v11 = sub_3988770(a2);
  sub_39A6760(a1, a3, v11, 73);
  if ( (*(_BYTE *)(*(_QWORD *)a2 + 36LL) & 0x40) != 0 )
    return sub_39A34D0((__int64)a1, a3, 52);
  result = sub_3988770(a2);
  if ( (*(_BYTE *)(result + 28) & 0x40) != 0 )
    return sub_39A34D0((__int64)a1, a3, 52);
  return result;
}
