// Function: sub_2C14320
// Address: 0x2c14320
//
unsigned __int64 __fastcall sub_2C14320(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int64 result; // rax
  int v10; // [rsp+0h] [rbp-50h]
  unsigned int v11; // [rsp+4h] [rbp-4Ch]
  __int64 v12; // [rsp+18h] [rbp-38h]

  v10 = *(_DWORD *)(*(_QWORD *)(a1 + 152) + 40LL);
  v4 = sub_2BFD6A0(a3 + 16, a1 + 96);
  v5 = sub_2AAEDF0(v4, a2);
  v11 = sub_1022EF0(*(_DWORD *)(*(_QWORD *)(a1 + 152) + 40LL));
  v6 = sub_DFD800(*(_QWORD *)a3, v11, v4, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
  if ( (unsigned int)(v10 - 6) <= 3 || (unsigned int)(v10 - 12) <= 3 )
  {
    sub_F6F040(v10);
    v7 = sub_DFDC70(*(_QWORD *)a3);
  }
  else
  {
    BYTE4(v12) = 1;
    LODWORD(v12) = *(_DWORD *)(*(_QWORD *)(a1 + 152) + 44LL);
    v7 = sub_DFDC10(*(__int64 **)a3, v11, v5, v12);
  }
  v8 = v7;
  result = v7 + v6;
  if ( __OFADD__(v8, v6) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v8 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
