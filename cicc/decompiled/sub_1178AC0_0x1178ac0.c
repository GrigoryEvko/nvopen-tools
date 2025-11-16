// Function: sub_1178AC0
// Address: 0x1178ac0
//
__int64 __fastcall sub_1178AC0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // r13
  __int64 v4; // r15
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // esi
  __int64 v10; // [rsp+8h] [rbp-98h]
  _BYTE v11[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v12; // [rsp+30h] [rbp-70h]
  _BYTE v13[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v14; // [rsp+60h] [rbp-40h]

  v2 = a1[1];
  v3 = *a1;
  v12 = 257;
  v10 = sub_AD62B0(*(_QWORD *)(*v2 + 8));
  v4 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v3[10] + 32LL))(
         v3[10],
         13,
         a2,
         v10,
         0,
         0);
  if ( !v4 )
  {
    v14 = 257;
    v4 = sub_B504D0(13, a2, v10, (__int64)v13, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v3[11] + 16LL))(
      v3[11],
      v4,
      v11,
      v3[7],
      v3[8]);
    v6 = *v3;
    v7 = *v3 + 16LL * *((unsigned int *)v3 + 2);
    while ( v7 != v6 )
    {
      v8 = *(_QWORD *)(v6 + 8);
      v9 = *(_DWORD *)v6;
      v6 += 16;
      sub_B99FD0(v4, v9, v8);
    }
  }
  v14 = 257;
  return sub_B504D0(28, *a1[2], v4, (__int64)v13, 0, 0);
}
