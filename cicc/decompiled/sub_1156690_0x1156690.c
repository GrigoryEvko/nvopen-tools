// Function: sub_1156690
// Address: 0x1156690
//
__int64 __fastcall sub_1156690(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // esi
  _BYTE v11[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v12; // [rsp+20h] [rbp-40h]

  v12 = 257;
  v4 = sub_BD2C40(72, unk_3F10A14);
  v5 = (__int64)v4;
  if ( v4 )
    sub_B549F0((__int64)v4, a2, (__int64)v11, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v5,
    a3,
    a1[7],
    a1[8]);
  v6 = *a1;
  v7 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v6 + 8);
      v9 = *(_DWORD *)v6;
      v6 += 16;
      sub_B99FD0(v5, v9, v8);
    }
    while ( v7 != v6 );
  }
  return v5;
}
