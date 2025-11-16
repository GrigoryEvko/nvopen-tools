// Function: sub_EBC0E0
// Address: 0xebc0e0
//
__int64 __fastcall sub_EBC0E0(__int64 *a1)
{
  __int64 v2; // rdi
  unsigned int v3; // r12d
  __int64 v4; // rax
  unsigned int v5; // ebx
  __int64 v6; // r14
  void (__fastcall *v7)(__int64, _QWORD *, _QWORD); // r15
  unsigned int v9; // [rsp+Ch] [rbp-44h]
  _QWORD *v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]

  v2 = *a1;
  v11 = 1;
  v10 = 0;
  if ( !*(_BYTE *)(v2 + 869) )
  {
    if ( (unsigned __int8)sub_EA2540(v2) )
    {
LABEL_8:
      v3 = 1;
      goto LABEL_9;
    }
    v2 = *a1;
  }
  v3 = sub_EBB490(v2, a1[1], (__int64)&v10);
  if ( (_BYTE)v3 )
    goto LABEL_8;
  v4 = *a1;
  v5 = v11;
  v6 = *(_QWORD *)(v4 + 232);
  v7 = *(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v6 + 536LL);
  if ( v11 <= 0x40 )
  {
    v7(v6, v10, v11 >> 3);
  }
  else
  {
    v9 = v11 >> 3;
    if ( v5 - (unsigned int)sub_C444A0((__int64)&v10) <= 0x40 )
      v7(v6, (_QWORD *)*v10, v9);
    else
      v7(v6, (_QWORD *)-1LL, v9);
  }
LABEL_9:
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return v3;
}
