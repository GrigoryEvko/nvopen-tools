// Function: sub_2A8B340
// Address: 0x2a8b340
//
__int64 __fastcall sub_2A8B340(__int64 a1)
{
  __int64 v1; // r14
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // r15
  bool v5; // cc
  unsigned __int64 v6; // rdi
  const void *v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v10; // [rsp+8h] [rbp-58h]
  __int64 v11; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v12; // [rsp+20h] [rbp-40h]

  v1 = a1 - 16;
  v2 = *(_DWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 16) = 0;
  v4 = *(_QWORD *)a1;
  v12 = v2;
  v10 = v3;
  v11 = v3;
  while ( v2 > 0x40 )
  {
    if ( sub_C43C50((__int64)&v11, (const void **)v1) )
      goto LABEL_10;
LABEL_3:
    if ( (int)sub_C4C880((__int64)&v11, v1) >= 0 )
      goto LABEL_11;
LABEL_4:
    v5 = *(_DWORD *)(v1 + 32) <= 0x40u;
    *(_QWORD *)(v1 + 16) = *(_QWORD *)(v1 - 8);
    if ( !v5 )
    {
      v6 = *(_QWORD *)(v1 + 24);
      if ( v6 )
        j_j___libc_free_0_0(v6);
    }
    v7 = *(const void **)v1;
    v1 -= 24;
    *(_QWORD *)(v1 + 48) = v7;
    LODWORD(v7) = *(_DWORD *)(v1 + 32);
    *(_DWORD *)(v1 + 32) = 0;
    *(_DWORD *)(v1 + 56) = (_DWORD)v7;
  }
  if ( v10 != *(_QWORD *)v1 )
    goto LABEL_3;
LABEL_10:
  if ( sub_B445A0(v4, *(_QWORD *)(v1 - 8)) )
    goto LABEL_4;
LABEL_11:
  v5 = *(_DWORD *)(v1 + 32) <= 0x40u;
  *(_QWORD *)(v1 + 16) = v4;
  if ( !v5 )
  {
    v8 = *(_QWORD *)(v1 + 24);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  *(_DWORD *)(v1 + 32) = v2;
  *(_QWORD *)(v1 + 24) = v10;
  return v10;
}
