// Function: sub_3914390
// Address: 0x3914390
//
__int64 __fastcall sub_3914390(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, char a5)
{
  unsigned int v8; // r13d
  __int64 v9; // rdi
  unsigned int v10; // eax
  unsigned __int32 v11; // ecx
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned __int32 v14; // ecx
  __int64 v15; // rdi
  unsigned int v16; // eax
  unsigned __int32 v17; // ecx
  unsigned int v18; // r9d
  __int64 v19; // rdi
  unsigned __int32 v20; // edx
  __int64 v21; // rdi
  unsigned __int32 v22; // edx
  __int64 v23; // rdi
  unsigned __int32 v24; // edx
  unsigned __int32 v25; // edx
  __int64 v26; // rdi
  __int64 result; // rax
  __int64 v28; // rdi
  char v29[52]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = (a5 != 0) << 13;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 240) + 64LL))(*(_QWORD *)(a1 + 240));
  v9 = *(_QWORD *)(a1 + 240);
  v10 = ((*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) & 1) != 0) - 17958194;
  v11 = _byteswap_ulong(v10);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v10 = v11;
  *(_DWORD *)v29 = v10;
  sub_16E7EE0(v9, v29, 4u);
  v12 = *(_QWORD *)(a1 + 240);
  v13 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 12LL);
  v14 = _byteswap_ulong(v13);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v13 = v14;
  *(_DWORD *)v29 = v13;
  sub_16E7EE0(v12, v29, 4u);
  v15 = *(_QWORD *)(a1 + 240);
  v16 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL);
  v17 = _byteswap_ulong(v16);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v16 = v17;
  *(_DWORD *)v29 = v16;
  sub_16E7EE0(v15, v29, 4u);
  v18 = a2;
  v19 = *(_QWORD *)(a1 + 240);
  v20 = _byteswap_ulong(a2);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v18 = v20;
  *(_DWORD *)v29 = v18;
  sub_16E7EE0(v19, v29, 4u);
  v21 = *(_QWORD *)(a1 + 240);
  v22 = _byteswap_ulong(a3);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a3 = v22;
  *(_DWORD *)v29 = a3;
  sub_16E7EE0(v21, v29, 4u);
  v23 = *(_QWORD *)(a1 + 240);
  v24 = _byteswap_ulong(a4);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    a4 = v24;
  *(_DWORD *)v29 = a4;
  sub_16E7EE0(v23, v29, 4u);
  v25 = _byteswap_ulong(v8);
  v26 = *(_QWORD *)(a1 + 240);
  if ( (unsigned int)(*(_DWORD *)(a1 + 248) - 1) > 1 )
    v8 = v25;
  *(_DWORD *)v29 = v8;
  sub_16E7EE0(v26, v29, 4u);
  result = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(result + 8) & 1) != 0 )
  {
    v28 = *(_QWORD *)(a1 + 240);
    *(_DWORD *)v29 = 0;
    return sub_16E7EE0(v28, v29, 4u);
  }
  return result;
}
