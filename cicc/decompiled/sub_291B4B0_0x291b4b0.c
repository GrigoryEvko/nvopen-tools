// Function: sub_291B4B0
// Address: 0x291b4b0
//
__int64 __fastcall sub_291B4B0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int16 a4, _BYTE *a5)
{
  char v6; // bl
  bool v7; // zf
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v15; // rax
  _BYTE *v16; // [rsp+0h] [rbp-90h] BYREF
  __int16 v17; // [rsp+20h] [rbp-70h]
  _BYTE v18[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  v6 = a4;
  v7 = *a5 == 0;
  v17 = 257;
  if ( !v7 )
  {
    v16 = a5;
    LOBYTE(v17) = 3;
  }
  if ( !HIBYTE(a4) )
  {
    v15 = sub_AA4E30(a1[6]);
    v6 = sub_AE5020(v15, a2);
  }
  v19 = 257;
  v8 = sub_BD2C40(80, 1u);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B4D190((__int64)v8, a2, a3, (__int64)v18, 0, v6, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    &v16,
    a1[7],
    a1[8]);
  v10 = *a1;
  v11 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v11 )
  {
    do
    {
      v12 = *(_QWORD *)(v10 + 8);
      v13 = *(_DWORD *)v10;
      v10 += 16;
      sub_B99FD0(v9, v13, v12);
    }
    while ( v11 != v10 );
  }
  return v9;
}
