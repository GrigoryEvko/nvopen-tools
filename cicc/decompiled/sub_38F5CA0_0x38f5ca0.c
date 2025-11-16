// Function: sub_38F5CA0
// Address: 0x38f5ca0
//
__int64 __fastcall sub_38F5CA0(
        __int64 *a1,
        __m128 a2,
        __m128 a3,
        double a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rdi
  unsigned int v11; // r12d
  __int64 v12; // rax
  unsigned int v13; // ebx
  __int64 v14; // r14
  void (__fastcall *v15)(__int64, _QWORD *, _QWORD); // r15
  unsigned int v17; // [rsp+Ch] [rbp-44h]
  _QWORD *v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  v10 = *a1;
  v19 = 1;
  v18 = 0;
  if ( !*(_BYTE *)(v10 + 845) )
  {
    if ( (unsigned __int8)sub_38E36C0(v10) )
    {
LABEL_8:
      v11 = 1;
      goto LABEL_9;
    }
    v10 = *a1;
  }
  v11 = sub_38F4CC0(v10, a1[1], (__int64)&v18, a2, a3, a4, a7, a8, a9);
  if ( (_BYTE)v11 )
    goto LABEL_8;
  v12 = *a1;
  v13 = v19;
  v14 = *(_QWORD *)(v12 + 328);
  v15 = *(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v14 + 424LL);
  if ( v19 <= 0x40 )
  {
    v15(v14, v18, v19 >> 3);
  }
  else
  {
    v17 = v19 >> 3;
    if ( v13 - (unsigned int)sub_16A57B0((__int64)&v18) <= 0x40 )
      v15(v14, (_QWORD *)*v18, v17);
    else
      v15(v14, (_QWORD *)-1LL, v17);
  }
LABEL_9:
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0((unsigned __int64)v18);
  return v11;
}
