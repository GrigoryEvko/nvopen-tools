// Function: sub_37385E0
// Address: 0x37385e0
//
void __fastcall sub_37385E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r10
  unsigned int v5; // eax
  __int64 v6; // rsi
  void (*v7)(); // rax
  void (*v8)(); // rax
  __int64 v9; // [rsp+8h] [rbp-8h] BYREF

  v4 = (_QWORD *)a1[23];
  v9 = a4;
  v5 = *(_DWORD *)(v4[25] + 544LL) - 42;
  if ( *(_BYTE *)(a2 + 88) != 1 )
  {
    if ( v5 <= 1 && *(_DWORD *)(a1[26] + 6224) == 1 )
      goto LABEL_13;
LABEL_8:
    sub_3738310(a1, a3, 2, (__int64)&v9);
    return;
  }
  if ( v5 > 1 || *(_DWORD *)(a1[26] + 6224) != 1 )
  {
    v6 = *(_QWORD *)(a2 + 48);
    if ( v6 )
    {
      sub_3738450(a1, v6, a3, 2, (__int64)&v9);
      return;
    }
    goto LABEL_8;
  }
  if ( *(_QWORD *)(a2 + 48) )
    goto LABEL_11;
LABEL_13:
  if ( !(_BYTE)v9 )
  {
LABEL_11:
    v7 = *(void (**)())(*v4 + 488LL);
    if ( v7 != nullsub_1833 )
      ((void (__fastcall *)(_QWORD *, __int64 *, __int64, __int64, __int64 *))v7)(v4, a1, a2, a3, &v9);
    return;
  }
  v8 = *(void (**)())(*v4 + 496LL);
  if ( v8 != nullsub_1834 )
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64, __int64 *))v8)(v4, a1, a3, &v9);
}
