// Function: sub_EABDC0
// Address: 0xeabdc0
//
__int64 __fastcall sub_EABDC0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  int *v4; // rdx
  int v5; // eax
  unsigned __int64 v6; // rcx
  _DWORD *v7; // rbx
  int v8; // eax
  unsigned __int64 v9; // r12
  __m128i v10; // xmm0
  bool v11; // cc
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  int *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  _BYTE v22[24]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v23; // [rsp+18h] [rbp-48h]
  unsigned int v24; // [rsp+20h] [rbp-40h]

  v2 = sub_ECD7B0(a1);
  v3 = sub_ECD6A0(v2);
LABEL_2:
  v4 = *(int **)(a1 + 48);
  v5 = *v4;
  if ( *v4 )
  {
    while ( v5 != 9 )
    {
      v6 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = 0;
      v7 = v4 + 10;
      v8 = v6;
      v6 *= 40LL;
      v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v6 - 40) >> 3);
      if ( v6 > 0x28 )
      {
        do
        {
          v10 = _mm_loadu_si128((const __m128i *)(v7 + 2));
          v11 = *(v7 - 2) <= 0x40u;
          *(v7 - 10) = *v7;
          *((__m128i *)v7 - 2) = v10;
          if ( !v11 )
          {
            v12 = *((_QWORD *)v7 - 2);
            if ( v12 )
              j_j___libc_free_0_0(v12);
          }
          v13 = *((_QWORD *)v7 + 3);
          v7 += 10;
          *((_QWORD *)v7 - 7) = v13;
          LODWORD(v13) = *(v7 - 2);
          *(v7 - 2) = 0;
          *(v7 - 12) = v13;
          --v9;
        }
        while ( v9 );
        v8 = *(_DWORD *)(a1 + 56);
        v4 = *(int **)(a1 + 48);
      }
      v14 = (unsigned int)(v8 - 1);
      *(_DWORD *)(a1 + 56) = v14;
      v15 = &v4[10 * v14];
      if ( (unsigned int)v15[8] > 0x40 )
      {
        v16 = *((_QWORD *)v15 + 3);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( *(_DWORD *)(a1 + 56) )
        goto LABEL_2;
      sub_1097F60(v22, a1 + 40);
      sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)v22, v17, v18, v19);
      if ( v24 <= 0x40 || !v23 )
        goto LABEL_2;
      j_j___libc_free_0_0(v23);
      v4 = *(int **)(a1 + 48);
      v5 = *v4;
      if ( !*v4 )
        break;
    }
  }
  v20 = sub_ECD7B0(a1);
  sub_ECD6A0(v20);
  return v3;
}
