// Function: sub_F94450
// Address: 0xf94450
//
__int64 __fastcall sub_F94450(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v9; // rax
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // esi
  _BYTE v17[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v18 = 257;
  v9 = sub_BD2C40(72, 3u);
  v10 = (__int64)v9;
  if ( v9 )
    sub_B4C9A0((__int64)v9, a3, a4, a2, 3u, 0, 0, 0);
  if ( a5 )
    sub_B99FD0(v10, 2u, a5);
  if ( a6 )
    sub_B99FD0(v10, 0xFu, a6);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v10,
    v17,
    a1[7],
    a1[8]);
  v11 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v11 )
  {
    v12 = *a1;
    do
    {
      v13 = *(_QWORD *)(v12 + 8);
      v14 = *(_DWORD *)v12;
      v12 += 16;
      sub_B99FD0(v10, v14, v13);
    }
    while ( v11 != v12 );
  }
  return v10;
}
