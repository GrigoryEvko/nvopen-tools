// Function: sub_32197E0
// Address: 0x32197e0
//
__int64 __fastcall sub_32197E0(__int64 a1, __int64 a2)
{
  _QWORD **v3; // rdi
  __int64 v4; // r12
  void (*v5)(); // rax
  _BYTE v7[32]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+20h] [rbp-20h]

  v3 = *(_QWORD ***)(a1 + 8);
  v4 = *(unsigned int *)(a2 + 16);
  v8 = 257;
  v5 = *(void (**)())(*v3[28] + 120LL);
  if ( v5 != nullsub_98 )
  {
    ((void (__fastcall *)(_QWORD *, _BYTE *, __int64))v5)(v3[28], v7, 1);
    v3 = *(_QWORD ***)(a1 + 8);
  }
  ((void (__fastcall *)(_QWORD **, __int64, _QWORD, __int64))(*v3)[53])(v3, v4, 0, 4);
  return 4;
}
