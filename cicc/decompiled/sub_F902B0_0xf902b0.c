// Function: sub_F902B0
// Address: 0xf902b0
//
void __fastcall sub_F902B0(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdx
  unsigned int v7; // esi
  _BYTE v8[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  v9 = 257;
  v2 = sub_BD2C40(72, 1u);
  v3 = (__int64)v2;
  if ( v2 )
    sub_B4C8F0((__int64)v2, a2, 1u, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v3,
    v8,
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
}
