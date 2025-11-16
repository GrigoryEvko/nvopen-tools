// Function: sub_1E41020
// Address: 0x1e41020
//
__int64 __fastcall sub_1E41020(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  unsigned int v12; // r14d
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  unsigned int v20; // [rsp+0h] [rbp-40h] BYREF
  int v21; // [rsp+4h] [rbp-3Ch] BYREF
  _BYTE v22[56]; // [rsp+8h] [rbp-38h] BYREF

  v4 = 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 112LL);
  if ( v9 != sub_1D00B10 )
    v4 = ((__int64 (__fastcall *)(__int64, __int64, int *, __int64, _QWORD))v9)(v8, a2, a3, a4, 0);
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 592LL);
  if ( v11 == sub_1D9BA90 )
    return 0;
  v12 = ((__int64 (__fastcall *)(__int64, __int64, unsigned int *, _BYTE *, __int64))v11)(v10, a2, &v20, v22, v4);
  if ( !(_BYTE)v12 )
    return 0;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
  v15 = sub_1E69D00(v14, v20);
  v16 = v15;
  if ( !v15 )
    return 0;
  v17 = **(unsigned __int16 **)(v15 + 16);
  if ( !v17 || v17 == 45 )
  {
    v20 = sub_1E40FE0(*(_QWORD *)(v16 + 32), *(_DWORD *)(v16 + 40), *(_QWORD *)(a2 + 24));
    v16 = sub_1E69D00(v14, v20);
    if ( !v16 )
      return 0;
  }
  v18 = *(_QWORD *)(a1 + 16);
  v21 = 0;
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 608LL);
  if ( v19 != sub_1E40460 && (((unsigned __int8 (__fastcall *)(__int64, __int64, int *))v19)(v18, v16, &v21) || v21 < 0) )
    *a3 = v21;
  else
    return 0;
  return v12;
}
