// Function: sub_39AEF10
// Address: 0x39aef10
//
__int64 __fastcall sub_39AEF10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 (__fastcall *v16)(__int64, __int64, unsigned __int64); // rbx
  unsigned __int64 v17; // rax
  _BYTE v20[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = a3;
  v7 = *(_DWORD *)(a2 + 712);
  if ( v7 == 0x7FFFFFFF )
  {
    v12 = 0;
  }
  else
  {
    v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL) + 16LL);
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 48LL);
    if ( v9 == sub_1D90020 )
      BUG();
    v10 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v9)(v8, a2, a3, a4, a3);
    v11 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _BYTE *))(*(_QWORD *)v10 + 176LL))(
            v10,
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL),
            v7,
            v20);
    v4 = a3;
    v12 = v11;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
  v14 = sub_38BF800(v13, v4, a4);
  v15 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64))(*(_QWORD *)v15 + 240LL);
  v17 = sub_38CB470(v12, v13);
  return v16(v15, v14, v17);
}
