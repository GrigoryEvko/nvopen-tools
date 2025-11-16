// Function: sub_11557E0
// Address: 0x11557e0
//
__int64 __fastcall sub_11557E0(__int64 a1, __int64 a2, _QWORD *a3, char a4)
{
  unsigned int v7; // r15d
  int v8; // eax
  unsigned int v9; // r8d
  unsigned int v10; // edx
  __int64 v11; // rdi
  unsigned int v12; // ebx
  __int64 v13; // r12
  unsigned int v14; // ebx
  int v15; // eax
  int v17; // eax
  int v18; // eax
  bool v19; // [rsp+Ch] [rbp-44h]
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  unsigned int v21; // [rsp+Ch] [rbp-44h]
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  if ( v7 <= 0x40 )
  {
    v9 = 0;
    if ( !*(_QWORD *)a2 )
      return v9;
    v10 = *(_DWORD *)(a1 + 8);
    if ( a4 )
      goto LABEL_4;
LABEL_9:
    v23 = v10;
    if ( v10 > 0x40 )
      sub_C43690((__int64)&v22, 0, 0);
    else
      v22 = 0;
    sub_C4BFE0(a1, a2, a3, &v22);
    goto LABEL_12;
  }
  v8 = sub_C444A0(a2);
  v9 = 0;
  if ( v7 == v8 )
    return v9;
  v10 = *(_DWORD *)(a1 + 8);
  if ( !a4 )
    goto LABEL_9;
LABEL_4:
  v11 = *(_QWORD *)a1;
  v12 = v10 - 1;
  if ( v10 <= 0x40 )
  {
    if ( v11 != 1LL << v12 )
    {
      v23 = v10;
LABEL_24:
      v22 = 0;
      goto LABEL_8;
    }
  }
  else if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) == 0
         || (v20 = v10, v17 = sub_C44590(a1), v10 = v20, v17 != v12) )
  {
    v23 = v10;
    goto LABEL_7;
  }
  v9 = 0;
  if ( !v7 )
    return v9;
  if ( v7 <= 0x40 )
  {
    if ( *(_QWORD *)a2 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) )
      return v9;
  }
  else
  {
    v21 = v10;
    v18 = sub_C445E0(a2);
    v10 = v21;
    v9 = 0;
    if ( v7 == v18 )
      return v9;
  }
  v23 = v10;
  if ( v10 <= 0x40 )
    goto LABEL_24;
LABEL_7:
  sub_C43690((__int64)&v22, 0, 1);
LABEL_8:
  sub_C4C400(a1, a2, (__int64)a3, (__int64)&v22);
LABEL_12:
  v13 = v22;
  v14 = v23;
  LOBYTE(v9) = v22 == 0;
  if ( v23 > 0x40 )
  {
    v15 = sub_C444A0((__int64)&v22);
    LOBYTE(v9) = v14 == v15;
    if ( v13 )
    {
      v19 = v14 == v15;
      j_j___libc_free_0_0(v13);
      return v19;
    }
  }
  return v9;
}
