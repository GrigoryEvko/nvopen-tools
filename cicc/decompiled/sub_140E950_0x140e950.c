// Function: sub_140E950
// Address: 0x140e950
//
__int64 __fastcall sub_140E950(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  unsigned int v8; // r12d
  _QWORD *v9; // rdi
  unsigned int v10; // eax
  _QWORD *v13; // [rsp+10h] [rbp-F0h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-E8h]
  __int64 v15; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-D8h]
  __int64 v17; // [rsp+30h] [rbp-D0h]
  unsigned int v18; // [rsp+38h] [rbp-C8h]
  _BYTE v19[24]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v20; // [rsp+58h] [rbp-A8h]
  unsigned int v21; // [rsp+60h] [rbp-A0h]
  __int64 v22; // [rsp+70h] [rbp-90h]
  unsigned __int64 v23; // [rsp+78h] [rbp-88h]

  v7 = sub_16498A0(a1);
  sub_140B840((__int64)v19, a3, a4, v7, (unsigned __int16)a5 | (BYTE2(a5) << 16));
  sub_140E6D0((__int64)&v15, (__int64)v19, a1);
  if ( v16 <= 1 )
  {
    v10 = v18;
    v8 = 0;
  }
  else
  {
    v8 = 0;
    if ( v18 <= 1 )
      goto LABEL_9;
    sub_140AE80((__int64)&v13, &v15);
    if ( v14 <= 0x40 )
    {
      *a2 = v13;
    }
    else
    {
      v9 = v13;
      *a2 = *v13;
      j_j___libc_free_0_0(v9);
    }
    v10 = v18;
    v8 = 1;
  }
  if ( v10 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
LABEL_9:
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v23 != v22 )
    _libc_free(v23);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return v8;
}
