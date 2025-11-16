// Function: sub_1DC3CB0
// Address: 0x1dc3cb0
//
void __fastcall sub_1DC3CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rax
  __int128 v16; // [rsp-20h] [rbp-240h]
  __int64 v17; // [rsp+10h] [rbp-210h]
  __int64 *i; // [rsp+18h] [rbp-208h]
  __int64 v19; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-1D8h]
  _BYTE *v21; // [rsp+60h] [rbp-1C0h]
  __int64 v22; // [rsp+68h] [rbp-1B8h]
  _BYTE v23[432]; // [rsp+70h] [rbp-1B0h] BYREF

  v6 = *(__int64 **)(a1 + 136);
  v21 = v23;
  v22 = 0x1000000000LL;
  v7 = *(unsigned int *)(a1 + 144);
  v19 = 0;
  v20 = 0;
  for ( i = &v6[4 * v7]; i != v6; v6 += 4 )
  {
    v8 = v6[1];
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 16);
      v10 = v6[2];
      v11 = 16LL * *(unsigned int *)(*(_QWORD *)v8 + 48LL);
      v12 = (__int64 *)(v11 + *(_QWORD *)(v9 + 392));
      v13 = *v12;
      if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        v9 = v6[3];
        v14 = (__int64 *)(*(_QWORD *)(a1 + 96) + v11);
        v10 = v12[1];
        v14[1] = 0;
        *v14 = v9;
      }
      v15 = *v6;
      if ( *v6 != v19 && (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v17 = *v6;
        sub_1DB55F0((__int64)&v19);
        v15 = v17;
      }
      v19 = v15;
      *((_QWORD *)&v16 + 1) = v10;
      *(_QWORD *)&v16 = v13;
      sub_1DB8AC0((__int64)&v19, a2, v9, (__int64)v12, a5, a6, v16, v6[3]);
    }
  }
  *(_DWORD *)(a1 + 144) = 0;
  sub_1DB55F0((__int64)&v19);
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
}
