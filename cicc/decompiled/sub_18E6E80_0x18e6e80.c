// Function: sub_18E6E80
// Address: 0x18e6e80
//
void __fastcall sub_18E6E80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rsi
  __int64 v8; // r12
  __int64 i; // rbx
  __int64 v10; // rdx
  int v11; // eax
  int v12; // esi
  __int64 v13; // [rsp-190h] [rbp-190h]
  char *v14; // [rsp-178h] [rbp-178h] BYREF
  __int64 v15; // [rsp-170h] [rbp-170h]
  _BYTE v16[128]; // [rsp-168h] [rbp-168h] BYREF
  __int64 v17; // [rsp-E8h] [rbp-E8h]
  int v18; // [rsp-E0h] [rbp-E0h]
  _QWORD *v19; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v20; // [rsp-D0h] [rbp-D0h]
  _QWORD v21[17]; // [rsp-C8h] [rbp-C8h] BYREF
  int v22; // [rsp-40h] [rbp-40h]

  v6 = a2 - a1;
  if ( v6 > 160 )
  {
    v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 5);
    v8 = (__int64)(v7 - 2) / 2;
    v13 = v7;
    for ( i = a1 + 160 * v8; ; i -= 160 )
    {
      v12 = *(_DWORD *)(i + 8);
      v15 = 0x800000000LL;
      v14 = v16;
      if ( v12 )
      {
        sub_18E63F0((__int64)&v14, (char **)i, a3, a4, a5, a6);
        v10 = *(_QWORD *)(i + 144);
        v20 = 0x800000000LL;
        v11 = *(_DWORD *)(i + 152);
        v19 = v21;
        v17 = v10;
        v18 = v11;
        if ( (_DWORD)v15 )
        {
          sub_18E63F0((__int64)&v19, &v14, v10, (unsigned int)v15, a5, a6);
          v10 = v17;
          v11 = v18;
        }
      }
      else
      {
        v10 = *(_QWORD *)(i + 144);
        v11 = *(_DWORD *)(i + 152);
        v20 = 0x800000000LL;
        v17 = v10;
        v18 = v11;
        v19 = v21;
      }
      v21[16] = v10;
      v22 = v11;
      sub_18E6AF0(a1, v8, v13, (unsigned __int64)&v19, a5, a6);
      if ( v19 != v21 )
        _libc_free((unsigned __int64)v19);
      if ( !v8 )
        break;
      --v8;
      if ( v14 != v16 )
        _libc_free((unsigned __int64)v14);
    }
    if ( v14 != v16 )
      _libc_free((unsigned __int64)v14);
  }
}
