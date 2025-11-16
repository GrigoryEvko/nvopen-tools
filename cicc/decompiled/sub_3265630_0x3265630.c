// Function: sub_3265630
// Address: 0x3265630
//
__int64 __fastcall sub_3265630(__int64 a1, __int64 a2)
{
  const void **v2; // r12
  int v3; // eax
  int v5; // eax
  __int64 v6; // rax
  unsigned int v7; // edx
  unsigned int v8; // r8d
  int v9; // eax
  const void *v10; // rax
  int v11; // eax
  const void *v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-38h]
  const void *v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  v13 = 1;
  v12 = 0;
  if ( !a2 || (v3 = *(_DWORD *)(a2 + 24), v3 != 35) && v3 != 11 )
  {
    LODWORD(v2) = sub_33D1410(a2, &v12);
    if ( !(_BYTE)v2 )
      goto LABEL_5;
    v7 = v13;
    goto LABEL_13;
  }
  v6 = *(_QWORD *)(a2 + 96);
  v7 = *(_DWORD *)(v6 + 32);
  if ( v7 > 0x40 )
  {
    sub_C43990((__int64)&v12, v6 + 24);
    v7 = v13;
LABEL_13:
    v8 = *(_DWORD *)(a1 + 8);
    if ( v8 != v7 )
      goto LABEL_14;
    if ( v7 > 0x40 )
    {
      LOBYTE(v5) = sub_C43C50(a1, &v12);
      LODWORD(v2) = v5;
      goto LABEL_9;
    }
LABEL_21:
    LOBYTE(v2) = *(_QWORD *)a1 == (_QWORD)v12;
    return (unsigned int)v2;
  }
  v10 = *(const void **)(v6 + 24);
  v8 = *(_DWORD *)(a1 + 8);
  v13 = v7;
  v12 = v10;
  if ( v7 == v8 )
    goto LABEL_21;
LABEL_14:
  v2 = &v14;
  if ( v8 <= v7 )
  {
    sub_C449B0((__int64)&v14, (const void **)a1, v7);
    if ( v15 <= 0x40 )
    {
      LOBYTE(v2) = v14 == v12;
      goto LABEL_5;
    }
    LOBYTE(v11) = sub_C43C50((__int64)&v14, &v12);
    LODWORD(v2) = v11;
    goto LABEL_18;
  }
  sub_C449B0((__int64)&v14, &v12, v8);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    LOBYTE(v2) = *(_QWORD *)a1 == (_QWORD)v14;
  }
  else
  {
    LOBYTE(v9) = sub_C43C50(a1, &v14);
    LODWORD(v2) = v9;
  }
  if ( v15 > 0x40 )
  {
LABEL_18:
    if ( v14 )
      j_j___libc_free_0_0((unsigned __int64)v14);
  }
LABEL_5:
  if ( v13 <= 0x40 )
    return (unsigned int)v2;
LABEL_9:
  if ( !v12 )
    return (unsigned int)v2;
  j_j___libc_free_0_0((unsigned __int64)v12);
  return (unsigned int)v2;
}
