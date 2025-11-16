// Function: sub_24E5740
// Address: 0x24e5740
//
__int64 __fastcall sub_24E5740(__int64 *a1)
{
  __int64 v1; // r14
  _QWORD *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdx
  unsigned int v7; // esi
  _BYTE v9[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v10; // [rsp+20h] [rbp-30h]

  v1 = a1[9];
  v10 = 257;
  v2 = sub_BD2C40(72, 0);
  v3 = (__int64)v2;
  if ( v2 )
    sub_B4BB80((__int64)v2, v1, 0, 0, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v3,
    v9,
    a1[7],
    a1[8]);
  v4 = *a1;
  v5 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v4 + 8);
      v7 = *(_DWORD *)v4;
      v4 += 16;
      sub_B99FD0(v3, v7, v6);
    }
    while ( v5 != v4 );
  }
  return v3;
}
