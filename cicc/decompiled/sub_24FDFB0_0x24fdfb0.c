// Function: sub_24FDFB0
// Address: 0x24fdfb0
//
__int64 __fastcall sub_24FDFB0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r10
  int v5; // ecx
  unsigned int v6; // edi
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r12
  int v11; // ecx
  int v12; // eax
  int v13; // ebx
  unsigned __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-28h]

  v15 = sub_AE43F0(*a1, *(_QWORD *)(a2 + 8));
  if ( v15 > 0x40 )
    sub_C43690((__int64)&v14, 0, 0);
  else
    v14 = 0;
  sub_BD45C0((unsigned __int8 *)a2, *a1, (__int64)&v14, 1, 0, 0, 0, 0);
  v2 = a1[1];
  if ( v15 > 0x40 )
  {
    v3 = *(_QWORD *)v14;
  }
  else
  {
    v3 = 0;
    if ( v15 )
      v3 = (__int64)(v14 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
  }
  if ( (*(_BYTE *)(v2 + 8) & 1) != 0 )
  {
    v4 = v2 + 16;
    v5 = 3;
  }
  else
  {
    v11 = *(_DWORD *)(v2 + 24);
    v4 = *(_QWORD *)(v2 + 16);
    if ( !v11 )
    {
LABEL_20:
      v9 = 0;
      goto LABEL_10;
    }
    v5 = v11 - 1;
  }
  v6 = v5 & (37 * v3);
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( v3 != *v7 )
  {
    v12 = 1;
    while ( v8 != 0x7FFFFFFFFFFFFFFFLL )
    {
      v13 = v12 + 1;
      v6 = v5 & (v12 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == v3 )
        goto LABEL_9;
      v12 = v13;
    }
    goto LABEL_20;
  }
LABEL_9:
  v9 = v7[1];
LABEL_10:
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v9;
}
