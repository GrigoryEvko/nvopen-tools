// Function: sub_388BFE0
// Address: 0x388bfe0
//
__int64 __fastcall sub_388BFE0(__int64 a1, __m128i *a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  int v5; // eax
  _QWORD *v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // r12d
  _BYTE *v10[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v12; // [rsp+30h] [rbp-50h] BYREF
  __int64 v13; // [rsp+38h] [rbp-48h]
  _BYTE v14[64]; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(_BYTE **)(a1 + 72);
  v4 = *(_QWORD *)(a1 + 80);
  v10[0] = v11;
  sub_3887850((__int64 *)v10, v3, (__int64)&v3[v4]);
  v5 = sub_3887100(a1 + 8);
  v12 = v14;
  v6 = v14;
  v7 = 0;
  *(_DWORD *)(a1 + 64) = v5;
  v13 = 0;
  v14[0] = 0;
  if ( v5 != 3 )
    goto LABEL_2;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v8 = sub_388B0A0(a1, (unsigned __int64 *)&v12);
  if ( !(_BYTE)v8 )
  {
    v7 = v13;
    v6 = v12;
LABEL_2:
    v8 = 0;
    sub_1562A10(a2, v10[0], (__int64)v10[1], v6, v7);
  }
  if ( v12 != (_QWORD *)v14 )
    j_j___libc_free_0((unsigned __int64)v12);
  if ( (_QWORD *)v10[0] != v11 )
    j_j___libc_free_0((unsigned __int64)v10[0]);
  return v8;
}
