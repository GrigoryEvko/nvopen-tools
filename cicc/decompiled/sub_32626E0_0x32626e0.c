// Function: sub_32626E0
// Address: 0x32626e0
//
__int64 __fastcall sub_32626E0(unsigned int *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned int v6; // ebx
  _QWORD *v7; // r13
  _QWORD *v8; // r12
  _QWORD *v10; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-68h]
  unsigned __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  const void *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  _QWORD *v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-38h]

  v3 = *a3;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v11 = *(_DWORD *)(v4 + 32);
  if ( v11 > 0x40 )
    sub_C43780((__int64)&v10, (const void **)(v4 + 24));
  else
    v10 = *(_QWORD **)(v4 + 24);
  v5 = *(_QWORD *)(v3 + 96);
  v13 = *(_DWORD *)(v5 + 32);
  if ( v13 > 0x40 )
    sub_C43780((__int64)&v12, (const void **)(v5 + 24));
  else
    v12 = *(_QWORD *)(v5 + 24);
  sub_3260590((__int64)&v10, (__int64)&v12, 1);
  v17 = v11;
  if ( v11 > 0x40 )
    sub_C43780((__int64)&v16, (const void **)&v10);
  else
    v16 = v10;
  sub_C45EE0((__int64)&v16, (__int64 *)&v12);
  v6 = v17;
  v7 = v16;
  v17 = 0;
  v8 = (_QWORD *)*a1;
  v15 = v6;
  v14 = v16;
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0((__int64)&v14) <= 0x40 && (unsigned __int64)v8 > *v7 )
    {
      LODWORD(v8) = 1;
    }
    else
    {
      LODWORD(v8) = 0;
      if ( !v7 )
        goto LABEL_9;
    }
    j_j___libc_free_0_0((unsigned __int64)v7);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0((unsigned __int64)v16);
  }
  else
  {
    LOBYTE(v8) = v16 < v8;
  }
LABEL_9:
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0((unsigned __int64)v10);
  return (unsigned int)v8;
}
