// Function: sub_3265EC0
// Address: 0x3265ec0
//
bool __fastcall sub_3265EC0(unsigned int *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned int v6; // r12d
  unsigned __int64 v7; // r13
  bool result; // al
  unsigned int v9; // r12d
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  unsigned int v12; // r12d
  bool v13; // [rsp+Fh] [rbp-71h]
  bool v14; // [rsp+Fh] [rbp-71h]
  bool v15; // [rsp+Fh] [rbp-71h]
  _QWORD *v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v18; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-58h]
  unsigned __int64 v20; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-48h]
  _QWORD *v22; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-38h]

  v3 = *a3;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v17 = *(_DWORD *)(v4 + 32);
  if ( v17 > 0x40 )
    sub_C43780((__int64)&v16, (const void **)(v4 + 24));
  else
    v16 = *(_QWORD **)(v4 + 24);
  v5 = *(_QWORD *)(v3 + 96);
  v19 = *(_DWORD *)(v5 + 32);
  if ( v19 > 0x40 )
    sub_C43780((__int64)&v18, (const void **)(v5 + 24));
  else
    v18 = *(unsigned __int64 **)(v5 + 24);
  sub_3260590((__int64)&v16, (__int64)&v18, 1);
  v6 = v19;
  v7 = *a1 - *((_QWORD *)a1 + 1);
  if ( v19 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0((__int64)&v18) <= 0x40 )
    {
      result = 0;
      if ( v7 > *v18 )
        goto LABEL_17;
    }
  }
  else
  {
    result = 0;
    if ( v7 > (unsigned __int64)v18 )
      goto LABEL_17;
  }
  v23 = v17;
  if ( v17 > 0x40 )
    sub_C43780((__int64)&v22, (const void **)&v16);
  else
    v22 = v16;
  sub_C45EE0((__int64)&v22, (__int64 *)&v18);
  v9 = v23;
  v10 = (_QWORD *)*a1;
  v23 = 0;
  v11 = v22;
  v21 = v9;
  v20 = (unsigned __int64)v22;
  if ( v9 <= 0x40 )
  {
    result = v22 < v10;
  }
  else
  {
    v12 = v9 - sub_C444A0((__int64)&v20);
    result = 0;
    if ( v12 <= 0x40 )
      result = (unsigned __int64)v10 > *v11;
    if ( v20 )
    {
      v13 = result;
      j_j___libc_free_0_0(v20);
      result = v13;
      if ( v23 > 0x40 )
      {
        if ( v22 )
        {
          j_j___libc_free_0_0((unsigned __int64)v22);
          result = v13;
        }
      }
    }
  }
  v6 = v19;
LABEL_17:
  if ( v6 > 0x40 && v18 )
  {
    v14 = result;
    j_j___libc_free_0_0((unsigned __int64)v18);
    result = v14;
  }
  if ( v17 > 0x40 )
  {
    if ( v16 )
    {
      v15 = result;
      j_j___libc_free_0_0((unsigned __int64)v16);
      return v15;
    }
  }
  return result;
}
