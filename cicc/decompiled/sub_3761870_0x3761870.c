// Function: sub_3761870
// Address: 0x3761870
//
__int64 __fastcall sub_3761870(_QWORD *a1, unsigned __int64 a2, unsigned __int16 a3, __int64 a4, char a5)
{
  __int64 v6; // rax
  _QWORD *v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 i; // rbx
  __int64 v14; // rdx
  _BYTE *v15; // rax
  _BYTE *v16; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+8h] [rbp-B8h]
  _BYTE v18[176]; // [rsp+10h] [rbp-B0h] BYREF

  v6 = *(unsigned int *)(a2 + 24);
  v7 = (_QWORD *)*a1;
  if ( (unsigned int)v6 > 0x1F3 || (v8 = 0, a3) && *((_BYTE *)v7 + 500 * a3 + v6 + 6414) == 4 )
  {
    v10 = a1[1];
    v16 = v18;
    v17 = 0x800000000LL;
    v11 = *v7;
    if ( a5 )
      (*(void (__fastcall **)(_QWORD *, unsigned __int64, _BYTE **, __int64))(v11 + 2424))(v7, a2, &v16, v10);
    else
      (*(void (__fastcall **)(_QWORD *, unsigned __int64, _BYTE **, __int64))(v11 + 2408))(v7, a2, &v16, v10);
    v8 = 0;
    if ( (_DWORD)v17 )
    {
      v12 = (unsigned int)v17;
      for ( i = 0; i != v12; ++i )
      {
        v14 = (unsigned int)i;
        v15 = &v16[16 * i];
        sub_3760E70((__int64)a1, a2, v14, *(_QWORD *)v15, *((_QWORD *)v15 + 1));
      }
      v8 = 1;
    }
    if ( v16 != v18 )
      _libc_free((unsigned __int64)v16);
  }
  return v8;
}
