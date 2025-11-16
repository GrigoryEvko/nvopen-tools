// Function: sub_2463EC0
// Address: 0x2463ec0
//
__int64 __fastcall sub_2463EC0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int16 a4, char a5)
{
  char v5; // r14
  _QWORD *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v15; // rax
  char v16; // al
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+8h] [rbp-68h]
  char v19[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v5 = a4;
  if ( !HIBYTE(a4) )
  {
    v18 = a3;
    v15 = sub_AA4E30(a1[6]);
    v16 = sub_AE5020(v15, *(_QWORD *)(a2 + 8));
    a3 = v18;
    v5 = v16;
  }
  v17 = a3;
  v20 = 257;
  v7 = sub_BD2C40(80, unk_3F10A10);
  v9 = (__int64)v7;
  if ( v7 )
    sub_B4D3C0((__int64)v7, a2, v17, a5, v5, v8, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    v19,
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
