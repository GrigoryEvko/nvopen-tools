// Function: sub_3598FB0
// Address: 0x3598fb0
//
__int64 __fastcall sub_3598FB0(__int64 a1, __int64 a2, int *a3)
{
  __int64 v5; // rax
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  __int64 (*v14)(); // rax
  char v16; // [rsp+Bh] [rbp-45h] BYREF
  int v17; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v19[56]; // [rsp+18h] [rbp-38h] BYREF

  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL));
  v6 = sub_2FE0930(*(__int64 **)(a1 + 32), a2, &v18, (__int64)v19, (__int64)&v16, v5);
  if ( !(_BYTE)v6 )
    return 0;
  if ( v16 )
    return 0;
  v7 = v6;
  if ( *(_BYTE *)v18 )
    return 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v9 = sub_2EBEE10(v8, *(_DWORD *)(v18 + 8));
  v10 = v9;
  if ( !v9 )
    return 0;
  v11 = *(unsigned __int16 *)(v9 + 68);
  if ( !v11 || v11 == 68 )
  {
    v12 = sub_3598190(v10, *(_QWORD *)(a2 + 24));
    v10 = sub_2EBEE10(v8, v12);
    if ( !v10 )
      return 0;
  }
  v13 = *(_QWORD *)(a1 + 32);
  v17 = 0;
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 864LL);
  if ( v14 != sub_2FDC6E0
    && (((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, int *))v14)(v13, v10, &v17) || v17 < 0) )
  {
    *a3 = v17;
  }
  else
  {
    return 0;
  }
  return v7;
}
