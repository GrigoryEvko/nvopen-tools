// Function: sub_38EAF10
// Address: 0x38eaf10
//
__int64 __fastcall sub_38EAF10(__int64 a1)
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
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  int *v15; // rax
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  _BYTE v19[24]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v20; // [rsp+18h] [rbp-48h]
  unsigned int v21; // [rsp+20h] [rbp-40h]

  v2 = sub_3909460(a1);
  v3 = sub_39092A0(v2);
LABEL_2:
  v4 = *(int **)(a1 + 152);
  v5 = *v4;
  if ( *v4 )
  {
    while ( v5 != 9 )
    {
      v6 = *(unsigned int *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = 0;
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
        v8 = *(_DWORD *)(a1 + 160);
        v4 = *(int **)(a1 + 152);
      }
      v14 = (unsigned int)(v8 - 1);
      *(_DWORD *)(a1 + 160) = v14;
      v15 = &v4[10 * v14];
      if ( (unsigned int)v15[8] > 0x40 )
      {
        v16 = *((_QWORD *)v15 + 3);
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
      if ( *(_DWORD *)(a1 + 160) )
        goto LABEL_2;
      sub_392C2E0(v19, a1 + 144);
      sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)v19);
      if ( v21 <= 0x40 || !v20 )
        goto LABEL_2;
      j_j___libc_free_0_0(v20);
      v4 = *(int **)(a1 + 152);
      v5 = *v4;
      if ( !*v4 )
        break;
    }
  }
  v17 = sub_3909460(a1);
  sub_39092A0(v17);
  return v3;
}
