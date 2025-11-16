// Function: sub_2B160A0
// Address: 0x2b160a0
//
__int64 __fastcall sub_2B160A0(unsigned int **a1, __int64 a2)
{
  unsigned __int64 **v2; // r13
  __int64 v3; // rsi
  unsigned int v4; // eax
  unsigned __int64 v5; // r12
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rbx
  unsigned int *v9; // rax
  unsigned __int64 *v10; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h]
  unsigned __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-38h]
  unsigned __int64 v16; // [rsp+30h] [rbp-30h]
  unsigned int v17; // [rsp+38h] [rbp-28h]

  LODWORD(v2) = 1;
  if ( *(_BYTE *)a2 == 13 )
    return (unsigned int)v2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_9AC3E0((__int64)&v14, *(_QWORD *)(v3 + 32), *((_QWORD *)*a1 + 418), 0, 0, 0, 0, 1);
  v4 = v15;
  v13 = v15;
  if ( v15 <= 0x40 )
  {
    v5 = v14;
LABEL_6:
    v6 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v5);
    if ( !v4 )
      v6 = 0;
    v7 = *a1[1];
    goto LABEL_9;
  }
  v2 = &v12;
  sub_C43780((__int64)&v12, (const void **)&v14);
  v4 = v13;
  if ( v13 <= 0x40 )
  {
    v5 = (unsigned __int64)v12;
    goto LABEL_6;
  }
  sub_C43D10((__int64)&v12);
  LODWORD(v2) = v13;
  v6 = v12;
  v9 = a1[1];
  v11 = v13;
  v10 = v12;
  v7 = *v9;
  if ( v13 <= 0x40 )
  {
LABEL_9:
    LOBYTE(v2) = (unsigned __int64)v6 < v7;
    goto LABEL_10;
  }
  if ( (unsigned int)v2 - (unsigned int)sub_C444A0((__int64)&v10) <= 0x40 && v7 > *v6 )
  {
    LODWORD(v2) = 1;
LABEL_23:
    j_j___libc_free_0_0((unsigned __int64)v6);
    goto LABEL_10;
  }
  LODWORD(v2) = 0;
  if ( v6 )
    goto LABEL_23;
LABEL_10:
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return (unsigned int)v2;
}
