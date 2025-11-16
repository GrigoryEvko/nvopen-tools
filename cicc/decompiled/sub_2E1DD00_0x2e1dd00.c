// Function: sub_2E1DD00
// Address: 0x2e1dd00
//
void __fastcall sub_2E1DD00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int128 v15; // [rsp-20h] [rbp-240h]
  __int64 v16; // [rsp+10h] [rbp-210h]
  __int64 *i; // [rsp+18h] [rbp-208h]
  __int64 v18; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 v19; // [rsp+48h] [rbp-1D8h]
  _BYTE *v20; // [rsp+60h] [rbp-1C0h]
  __int64 v21; // [rsp+68h] [rbp-1B8h]
  _BYTE v22[432]; // [rsp+70h] [rbp-1B0h] BYREF

  v6 = *(__int64 **)(a1 + 184);
  v20 = v22;
  v21 = 0x1000000000LL;
  v7 = *(unsigned int *)(a1 + 192);
  v18 = 0;
  v19 = 0;
  for ( i = &v6[4 * v7]; i != v6; v6 += 4 )
  {
    v10 = v6[1];
    if ( v10 )
    {
      v14 = v6[2];
      v11 = 16LL * *(unsigned int *)(*(_QWORD *)v10 + 24LL);
      a2 = (__int64 *)(v11 + *(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL));
      v12 = *a2;
      v8 = a2[1];
      if ( (v14 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v8 = v6[2];
      }
      else
      {
        v14 = v6[3];
        v13 = (__int64 *)(*(_QWORD *)(a1 + 144) + v11);
        v13[1] = 0;
        *v13 = v14;
      }
      v9 = *v6;
      if ( *v6 != v18 && (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v16 = v8;
        sub_2E0B930((__int64)&v18, (__int64)a2, v8, v14, a5);
        v8 = v16;
      }
      v18 = v9;
      *((_QWORD *)&v15 + 1) = v8;
      *(_QWORD *)&v15 = v12;
      sub_2E0F380((__int64)&v18, (__int64)a2, v8, v14, a5, a6, v15, v6[3]);
    }
  }
  *(_DWORD *)(a1 + 192) = 0;
  sub_2E0B930((__int64)&v18, (__int64)a2, a3, a4, a5);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
}
