// Function: sub_2DAC5B0
// Address: 0x2dac5b0
//
__int64 __fastcall sub_2DAC5B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 (*v5)(); // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  void *v11; // rsi
  __int64 v13; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v14; // [rsp+8h] [rbp-88h]
  int v15; // [rsp+10h] [rbp-80h]
  int v16; // [rsp+14h] [rbp-7Ch]
  int v17; // [rsp+18h] [rbp-78h]
  char v18; // [rsp+1Ch] [rbp-74h]
  _QWORD v19[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v20; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v21; // [rsp+38h] [rbp-58h]
  __int64 v22; // [rsp+40h] [rbp-50h]
  int v23; // [rsp+48h] [rbp-48h]
  char v24; // [rsp+4Ch] [rbp-44h]
  _BYTE v25[64]; // [rsp+50h] [rbp-40h] BYREF

  v5 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v5 == sub_23CE270 )
    BUG();
  v8 = 0;
  v9 = ((__int64 (__fastcall *)(_QWORD, __int64))v5)(*a2, a3);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
  if ( v10 != sub_2C8F680 )
    v8 = ((__int64 (__fastcall *)(__int64))v10)(v9);
  v13 = v8;
  v14 = (_QWORD *)(sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8);
  v11 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2DAC510((__int64)&v13, a3) )
  {
    v14 = v19;
    v15 = 2;
    v19[0] = &unk_4F82418;
    v17 = 0;
    v18 = 1;
    v20 = 0;
    v21 = v25;
    v22 = 2;
    v23 = 0;
    v24 = 1;
    v16 = 1;
    v13 = 1;
    sub_C8CF70(a1, v11, 2, (__int64)v19, (__int64)&v13);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v25, (__int64)&v20);
    if ( v24 )
    {
      if ( v18 )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v21);
      if ( v18 )
        return a1;
    }
    _libc_free((unsigned __int64)v14);
    return a1;
  }
  *(_QWORD *)(a1 + 8) = v11;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  return a1;
}
