// Function: sub_20DA520
// Address: 0x20da520
//
__int64 __fastcall sub_20DA520(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  _QWORD *v3; // r15
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  __int64 v8; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+8h] [rbp-48h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  int v11; // [rsp+18h] [rbp-38h]

  v2 = 0;
  sub_1E0BDD0(a2, 0);
  v3 = a2 + 40;
  sub_20C9140((__int64)&v8, (__int64)a2);
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  v4 = v9;
  ++*(_QWORD *)(a1 + 80);
  ++v8;
  *(_QWORD *)(a1 + 88) = v4;
  v9 = 0;
  *(_QWORD *)(a1 + 96) = v10;
  v10 = 0;
  *(_DWORD *)(a1 + 104) = v11;
  v11 = 0;
  j___libc_free_0(0);
  v5 = *(_QWORD **)(a2[41] + 8LL);
  if ( a2 + 40 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5;
        v5 = (_QWORD *)v5[1];
        v2 |= sub_20D8B60(a1, (__int64)v6);
        if ( v6[9] == v6[8] )
          break;
        if ( v3 == v5 )
          return v2;
      }
      v2 = 1;
      sub_20D6E00(a1, v6);
    }
    while ( v3 != v5 );
  }
  return v2;
}
