// Function: sub_23DEB90
// Address: 0x23deb90
//
__int64 __fastcall sub_23DEB90(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  unsigned int v7; // ebx
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // esi
  unsigned __int8 v16; // [rsp+Fh] [rbp-61h]
  _BYTE v17[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v18; // [rsp+30h] [rbp-40h]

  v5 = sub_AA4E30(a1[6]);
  v6 = sub_AE5260(v5, (__int64)a2);
  v7 = *(_DWORD *)(v5 + 4);
  v16 = v6;
  v18 = 257;
  v8 = sub_BD2C40(80, unk_3F10A14);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B4CCA0((__int64)v8, a2, v7, a3, v16, (__int64)v17, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a4,
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
