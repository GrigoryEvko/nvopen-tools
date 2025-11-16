// Function: sub_123C800
// Address: 0x123c800
//
__int64 __fastcall sub_123C800(__int64 a1, unsigned int a2)
{
  unsigned int v3; // r15d
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+10h] [rbp-60h]
  __int64 v14; // [rsp+18h] [rbp-58h]
  _QWORD *v15; // [rsp+20h] [rbp-50h] BYREF
  size_t v16; // [rsp+28h] [rbp-48h]
  _QWORD v17[8]; // [rsp+30h] [rbp-40h] BYREF

  v15 = v17;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_120AFE0(a1, 413, "expected 'name' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120B3D0(a1, (__int64)&v15)
    || (v5 = sub_9CA790(*(_QWORD **)(a1 + 352), v15, v16), (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here"))
    || (unsigned __int8)sub_123C730(a1, v5)
    || (v3 = sub_120AFE0(a1, 13, "expected ')' here"), (_BYTE)v3) )
  {
    v3 = 1;
  }
  else
  {
    v12 = a1 + 1656;
    v6 = *(_QWORD *)(a1 + 1664);
    if ( v6 )
    {
      v14 = a1 + 1656;
      do
      {
        if ( *(_DWORD *)(v6 + 32) < a2 )
        {
          v6 = *(_QWORD *)(v6 + 24);
        }
        else
        {
          v14 = v6;
          v6 = *(_QWORD *)(v6 + 16);
        }
      }
      while ( v6 );
      if ( v14 != v12 && *(_DWORD *)(v14 + 32) <= a2 )
      {
        v7 = *(_QWORD *)(v14 + 40);
        for ( i = *(_QWORD *)(v14 + 48); i != v7; *v8 = sub_B2F650((__int64)v15, v16) )
        {
          v7 += 16;
          v8 = *(__int64 **)(v7 - 16);
        }
        v9 = sub_220F330(v14, v12);
        v10 = *(_QWORD *)(v9 + 40);
        v11 = v9;
        if ( v10 )
          j_j___libc_free_0(v10, *(_QWORD *)(v9 + 56) - v10);
        j_j___libc_free_0(v11, 64);
        --*(_QWORD *)(a1 + 1688);
      }
    }
  }
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  return v3;
}
