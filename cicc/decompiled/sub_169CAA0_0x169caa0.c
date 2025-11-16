// Function: sub_169CAA0
// Address: 0x169caa0
//
__int64 __fastcall sub_169CAA0(__int64 a1, unsigned __int8 a2, unsigned __int8 a3, __int64 *a4, float a5)
{
  unsigned int v6; // r14d
  void *v8; // rbx
  __int64 v9; // r8
  _BYTE *v10; // rdi
  void **v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v6 = a3;
  v13 = *(_QWORD *)(a1 + 8);
  v8 = sub_16982C0();
  v10 = (_BYTE *)(v13 + 8);
  if ( *(void **)(v13 + 8) == v8 )
    sub_169CAA0(v10, a2, v6, a4, v9, a5);
  else
    sub_16986F0(v10, a2, v6, a4);
  v11 = (void **)(*(_QWORD *)(a1 + 8) + 40LL);
  if ( *v11 == v8 )
    return sub_169C980(v11, 0);
  else
    return sub_169B620((__int64)v11, 0);
}
