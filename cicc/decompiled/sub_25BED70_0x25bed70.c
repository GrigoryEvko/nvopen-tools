// Function: sub_25BED70
// Address: 0x25bed70
//
__int64 __fastcall sub_25BED70(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  unsigned int v5; // ebx
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // eax
  bool v10; // cc
  unsigned __int64 v11; // [rsp+8h] [rbp-68h]
  unsigned __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  unsigned __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  int v17; // [rsp+38h] [rbp-38h]
  __int64 v18; // [rsp+40h] [rbp-30h]
  int v19; // [rsp+48h] [rbp-28h]

  if ( *(_BYTE *)a2 != 17 || !a4 )
    goto LABEL_3;
  v5 = *(_DWORD *)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 24);
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 <= 0x40 )
  {
    if ( (v7 & v6) == 0 && v6 )
    {
      v8 = a3;
      if ( v5 )
        v8 = ((__int64)(v6 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5)) + a3;
      goto LABEL_13;
    }
LABEL_3:
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  if ( (*(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6)) & v7) != 0 )
    goto LABEL_3;
  v11 = a3;
  if ( v5 == (unsigned int)sub_C444A0(a2 + 24) )
    goto LABEL_3;
  a3 = v11;
  v8 = v11 + *(_QWORD *)v6;
LABEL_13:
  v14 = a3;
  v12 = v8;
  v13 = 64;
  v15 = 64;
  sub_AADC30((__int64)&v16, (__int64)&v14, (__int64 *)&v12);
  v9 = v17;
  v10 = v15 <= 0x40;
  *(_BYTE *)(a1 + 32) = 1;
  *(_DWORD *)(a1 + 8) = v9;
  *(_QWORD *)a1 = v16;
  *(_DWORD *)(a1 + 24) = v19;
  *(_QWORD *)(a1 + 16) = v18;
  if ( !v10 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return a1;
}
