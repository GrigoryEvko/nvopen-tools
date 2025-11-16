// Function: sub_2E217F0
// Address: 0x2e217f0
//
__int64 __fastcall sub_2E217F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int16 *v10; // r12
  unsigned int v11; // r15d
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  unsigned __int64 v17; // r12
  __int128 v19; // [rsp-18h] [rbp-1F0h]
  __int64 v20; // [rsp+18h] [rbp-1C0h]
  __int64 v21; // [rsp+20h] [rbp-1B8h]
  __int64 v22; // [rsp+38h] [rbp-1A0h]
  __int64 v23; // [rsp+40h] [rbp-198h]
  int v24; // [rsp+58h] [rbp-180h] BYREF
  __int64 v25; // [rsp+60h] [rbp-178h]
  __int64 v26; // [rsp+68h] [rbp-170h]
  __int64 v27; // [rsp+70h] [rbp-168h]
  int *v28; // [rsp+78h] [rbp-160h]
  unsigned __int64 v29[2]; // [rsp+88h] [rbp-150h] BYREF
  _BYTE v30[48]; // [rsp+98h] [rbp-140h] BYREF
  _BYTE *v31; // [rsp+C8h] [rbp-110h]
  __int64 v32; // [rsp+D0h] [rbp-108h]
  _BYTE v33[16]; // [rsp+D8h] [rbp-100h] BYREF
  unsigned __int64 v34; // [rsp+E8h] [rbp-F0h]
  _QWORD v35[4]; // [rsp+F8h] [rbp-E0h] BYREF
  _BYTE *v36; // [rsp+118h] [rbp-C0h]
  __int64 v37; // [rsp+120h] [rbp-B8h]
  _BYTE v38[64]; // [rsp+128h] [rbp-B0h] BYREF
  _BYTE *v39; // [rsp+168h] [rbp-70h]
  __int64 v40; // [rsp+170h] [rbp-68h]
  _BYTE v41[32]; // [rsp+178h] [rbp-60h] BYREF
  __int16 v42; // [rsp+198h] [rbp-40h]
  unsigned __int64 v43; // [rsp+19Ch] [rbp-3Ch]

  v6 = a4;
  v27 = a3;
  v28 = &v24;
  *((_QWORD *)&v19 + 1) = a3;
  *(_QWORD *)&v19 = a2;
  v29[0] = (unsigned __int64)v30;
  v25 = a2;
  v26 = a2;
  v31 = v33;
  v24 = 0;
  v29[1] = 0x200000000LL;
  v32 = 0x200000000LL;
  v34 = 0;
  sub_2E0F080((__int64)v29, a2, a3, (__int64)v33, a5, a6, v19, (__int64)&v24);
  v20 = 0;
  v21 = 0;
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 8LL) + 24 * v6;
  v10 = (__int16 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 2LL * (*(_DWORD *)(v9 + 16) >> 12));
  v11 = *(_DWORD *)(v9 + 16) & 0xFFF;
  v12 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 64LL) + 16LL * *(unsigned __int16 *)(v9 + 20));
  if ( v10 )
  {
    do
    {
      v23 = *v12;
      v13 = v12[1];
      v43 = 0;
      v22 = v13;
      v35[3] = 0;
      v36 = v38;
      v37 = 0x400000000LL;
      v39 = v41;
      v42 = 0;
      v40 = 0x400000000LL;
      v14 = *(_QWORD *)(a1 + 48);
      v35[1] = v29;
      v15 = *(unsigned int *)(a1 + 24);
      v35[0] = v14 + 216LL * v11;
      v43 = __PAIR64__(v15, *(_DWORD *)v35[0]);
      if ( (unsigned int)sub_2E1AC90((__int64)v35, 1u, v15, (__int64)v29, v7, v8) )
      {
        v21 |= v23;
        v20 |= v22;
      }
      if ( v39 != v41 )
        _libc_free((unsigned __int64)v39);
      if ( v36 != v38 )
        _libc_free((unsigned __int64)v36);
      v16 = *v10;
      v12 += 2;
      ++v10;
      v11 += v16;
    }
    while ( (_WORD)v16 );
  }
  v17 = v34;
  if ( v34 )
  {
    sub_2E20A30(*(_QWORD *)(v34 + 16));
    j_j___libc_free_0(v17);
  }
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0]);
  return v21;
}
