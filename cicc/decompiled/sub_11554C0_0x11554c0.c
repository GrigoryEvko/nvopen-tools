// Function: sub_11554C0
// Address: 0x11554c0
//
__int64 __fastcall sub_11554C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v18; // [rsp+18h] [rbp-B8h]
  __int64 v19; // [rsp+28h] [rbp-A8h]
  _QWORD v20[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v21[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v22; // [rsp+60h] [rbp-70h]
  _BYTE v23[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v24; // [rsp+90h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 80);
  v22 = 257;
  v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 32LL))(
         v8,
         13,
         a4,
         a5,
         0,
         0);
  if ( !v9 )
  {
    v24 = 257;
    v9 = sub_B504D0(13, a4, a5, (__int64)v23, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v9,
      v21,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v18 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v18 )
    {
      v13 = *(_QWORD *)a2;
      do
      {
        v14 = *(_QWORD *)(v13 + 8);
        v15 = *(_DWORD *)v13;
        v13 += 16;
        sub_B99FD0(v9, v15, v14);
      }
      while ( v18 != v13 );
    }
  }
  v24 = 257;
  LODWORD(v19) = sub_B45210(a1);
  v10 = *(_QWORD *)(a3 + 8);
  v20[1] = v9;
  BYTE4(v19) = 1;
  v21[0] = v10;
  v11 = *(_QWORD *)(v9 + 8);
  v20[0] = a3;
  v21[1] = v11;
  return sub_B33D10(a2, 0x11Du, (__int64)v21, 2, (int)v20, 2, v19, (__int64)v23);
}
