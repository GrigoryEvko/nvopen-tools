// Function: sub_1D0B420
// Address: 0x1d0b420
//
__int64 __fastcall sub_1D0B420(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v5; // r9
  unsigned int v9; // r8d
  unsigned int *v10; // rax
  unsigned int v11; // eax
  _BYTE *v12; // r11
  int v13; // r8d
  size_t v14; // r10
  __int64 v15; // rcx
  _BYTE *v17; // rdi
  _QWORD *v18; // rcx
  __int64 v20; // [rsp-A8h] [rbp-A8h]
  _BYTE *v21; // [rsp-A0h] [rbp-A0h]
  int v22; // [rsp-A0h] [rbp-A0h]
  unsigned int v23; // [rsp-98h] [rbp-98h]
  __int64 v24; // [rsp-98h] [rbp-98h]
  __int64 v25; // [rsp-98h] [rbp-98h]
  _BYTE *v26; // [rsp-88h] [rbp-88h] BYREF
  __int64 v27; // [rsp-80h] [rbp-80h]
  _BYTE v28[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a1 != a2 )
  {
    v5 = a3;
    if ( a2 )
    {
      v9 = 0;
      v10 = (unsigned int *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1));
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v10 + 40LL) + 16LL * v10[2]) == 111 )
        return v9;
    }
    v11 = *(_DWORD *)(a1 + 60);
    v12 = *(_BYTE **)(a1 + 40);
    v9 = 0;
    if ( v12[16 * v11 - 16] == 111 )
      return v9;
    v13 = *(_DWORD *)(a1 + 60);
    v26 = v28;
    v14 = 16LL * v11;
    v27 = 0x400000000LL;
    if ( v11 > 4uLL )
    {
      v20 = 16LL * v11;
      v21 = v12;
      v23 = v11;
      sub_16CD150((__int64)&v26, v28, v11, 16, v11, a3);
      v13 = v23;
      v12 = v21;
      v14 = v20;
      v5 = a3;
      v17 = &v26[16 * (unsigned int)v27];
    }
    else
    {
      if ( !v14 )
      {
LABEL_7:
        LODWORD(v27) = v13 + v14;
        v15 = (unsigned int)(v13 + v14);
        if ( a4 )
        {
          if ( (unsigned int)(v13 + v14) >= HIDWORD(v27) )
          {
            v25 = v5;
            sub_16CD150((__int64)&v26, v28, 0, 16, v13, v5);
            v15 = (unsigned int)v27;
            v5 = v25;
          }
          v18 = &v26[16 * v15];
          *v18 = 111;
          v18[1] = 0;
          v15 = (unsigned int)(v27 + 1);
          LODWORD(v27) = v27 + 1;
        }
        sub_1D0B1B0(a1, a5, (__int64)v26, v15, a2, v5);
        if ( v26 != v28 )
          _libc_free((unsigned __int64)v26);
        return 1;
      }
      v17 = v28;
    }
    v22 = v13;
    v24 = v5;
    memcpy(v17, v12, v14);
    LODWORD(v14) = v27;
    v13 = v22;
    v5 = v24;
    goto LABEL_7;
  }
  return 0;
}
