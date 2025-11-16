// Function: sub_39C9A70
// Address: 0x39c9a70
//
void __fastcall sub_39C9A70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _DWORD *v7; // rdi
  void (*v8)(); // rax
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1[24];
  v9[0] = a4;
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v6 + 232) + 504LL) - 34) <= 1 && (v7 = (_DWORD *)a1[25], v7[1646] == 1) )
  {
    v8 = *(void (**)())(*(_QWORD *)v7 + 136LL);
    if ( v8 != nullsub_1983 )
      ((void (__fastcall *)(_DWORD *, __int64 *, __int64, __int64, _QWORD))v8)(v7, a1, a2, a3, v9[0]);
  }
  else if ( *(_DWORD *)(a2 + 48) )
  {
    sub_39C98B0(a1, a2, a3, 2, (__int64)v9);
  }
  else if ( (unsigned __int8)sub_3988750(a2) )
  {
    sub_39A4610(a1, a2, a3, 2, (__int64)v9);
  }
  else
  {
    sub_39C9790(a1, a3, 2, (__int64)v9);
  }
}
