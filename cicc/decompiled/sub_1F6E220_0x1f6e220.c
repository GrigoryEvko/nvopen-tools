// Function: sub_1F6E220
// Address: 0x1f6e220
//
__int64 __fastcall sub_1F6E220(unsigned int *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned int v6; // ebx
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 *v10; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-68h]
  __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-38h]

  v3 = *a3;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  v11 = *(_DWORD *)(v4 + 32);
  if ( v11 > 0x40 )
    sub_16A4FD0((__int64)&v10, (const void **)(v4 + 24));
  else
    v10 = *(unsigned __int64 **)(v4 + 24);
  v5 = *(_QWORD *)(v3 + 88);
  v13 = *(_DWORD *)(v5 + 32);
  if ( v13 > 0x40 )
    sub_16A4FD0((__int64)&v12, (const void **)(v5 + 24));
  else
    v12 = *(_QWORD *)(v5 + 24);
  sub_1F6DAA0((__int64)&v10, (__int64)&v12, 1);
  v17 = v11;
  if ( v11 > 0x40 )
    sub_16A4FD0((__int64)&v16, (const void **)&v10);
  else
    v16 = v10;
  sub_16A7200((__int64)&v16, &v12);
  v6 = v17;
  v7 = v16;
  v17 = 0;
  v8 = *a1;
  v15 = v6;
  v14 = v16;
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_16A57B0((__int64)&v14) <= 0x40 && v8 > *v7 )
    {
      LODWORD(v8) = 1;
    }
    else
    {
      LODWORD(v8) = 0;
      if ( !v7 )
        goto LABEL_9;
    }
    j_j___libc_free_0_0(v7);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
  }
  else
  {
    LOBYTE(v8) = (unsigned __int64)v16 < v8;
  }
LABEL_9:
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return (unsigned int)v8;
}
