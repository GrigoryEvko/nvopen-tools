// Function: sub_24E57E0
// Address: 0x24e57e0
//
__int64 __fastcall sub_24E57E0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r15
  unsigned int v4; // esi
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  unsigned int v10; // esi
  _BYTE v12[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v13; // [rsp+20h] [rbp-40h]

  v3 = a1[9];
  v13 = 257;
  v4 = a2 != 0;
  v5 = sub_BD2C40(72, v4);
  v6 = (__int64)v5;
  if ( v5 )
    sub_B4BB80((__int64)v5, v3, a2, v4, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v6,
    v12,
    a1[7],
    a1[8]);
  v7 = *a1;
  v8 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v7 + 8);
      v10 = *(_DWORD *)v7;
      v7 += 16;
      sub_B99FD0(v6, v10, v9);
    }
    while ( v8 != v7 );
  }
  return v6;
}
