// Function: sub_EB61F0
// Address: 0xeb61f0
//
__int64 __fastcall sub_EB61F0(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r13d
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r13
  int v14; // edx
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rax
  __m128i v17; // xmm0
  bool v18; // cc
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // [rsp+0h] [rbp-A0h]
  __int64 v29; // [rsp+8h] [rbp-98h]
  int v30; // [rsp+10h] [rbp-90h] BYREF
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+28h] [rbp-78h]
  unsigned int v34; // [rsp+30h] [rbp-70h]
  _BYTE v35[24]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v36; // [rsp+58h] [rbp-48h]
  unsigned int v37; // [rsp+60h] [rbp-40h]

  v4 = **(_DWORD **)(a1 + 48);
  LOBYTE(v2) = v4 == 27 || v4 == 46;
  if ( (_BYTE)v2 )
  {
    v10 = a1 + 40;
    v29 = sub_ECD690(a1 + 40);
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v34 = 1;
    v33 = 0;
    sub_1095550(a1 + 40, &v30, 1, 0);
    if ( ((v30 - 2) & 0xFFFFFFFD) == 0 && v29 + 1 == sub_ECD6A0(&v30) )
    {
      v12 = *(unsigned int *)(a1 + 56);
      v13 = *(_QWORD *)(a1 + 48);
      v14 = *(_DWORD *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = *(_DWORD *)v13 == 9;
      v15 = 40 * v12;
      v16 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v12 - 40) >> 3);
      if ( v15 > 0x28 )
      {
        do
        {
          v17 = _mm_loadu_si128((const __m128i *)(v13 + 48));
          v18 = *(_DWORD *)(v13 + 32) <= 0x40u;
          *(_DWORD *)v13 = *(_DWORD *)(v13 + 40);
          *(__m128i *)(v13 + 8) = v17;
          if ( !v18 )
          {
            v19 = *(_QWORD *)(v13 + 24);
            if ( v19 )
            {
              v28 = v16;
              j_j___libc_free_0_0(v19);
              v16 = v28;
            }
          }
          v20 = *(_QWORD *)(v13 + 64);
          v13 += 40;
          *(_QWORD *)(v13 - 16) = v20;
          LODWORD(v20) = *(_DWORD *)(v13 + 32);
          *(_DWORD *)(v13 + 32) = 0;
          *(_DWORD *)(v13 - 8) = v20;
          --v16;
        }
        while ( v16 );
        v14 = *(_DWORD *)(a1 + 56);
        v13 = *(_QWORD *)(a1 + 48);
      }
      v21 = (unsigned int)(v14 - 1);
      *(_DWORD *)(a1 + 56) = v21;
      v22 = v13 + 40 * v21;
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
      {
        v23 = *(_QWORD *)(v22 + 24);
        if ( v23 )
          j_j___libc_free_0_0(v23);
      }
      if ( !*(_DWORD *)(a1 + 56) )
      {
        sub_1097F60(v35, v10);
        sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)v35, v24, v25, v26);
        if ( v37 > 0x40 )
        {
          if ( v36 )
            j_j___libc_free_0_0(v36);
        }
      }
      v2 = 0;
      v27 = *(_QWORD *)(sub_ECD7B0(a1) + 16);
      *a2 = v29;
      a2[1] = v27 + 1;
      sub_EABFE0(a1);
    }
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
  }
  else if ( (unsigned int)(v4 - 2) > 1 )
  {
    return 1;
  }
  else
  {
    v5 = sub_ECD7B0(a1);
    if ( *(_DWORD *)v5 == 2 )
    {
      v11 = *(_QWORD *)(v5 + 16);
      *a2 = *(_QWORD *)(v5 + 8);
      a2[1] = v11;
      sub_EABFE0(a1);
    }
    else
    {
      v6 = *(_QWORD *)(v5 + 16);
      v7 = *(_QWORD *)(v5 + 8);
      if ( v6 )
      {
        v8 = v6 - 1;
        if ( !v8 )
          v8 = 1;
        ++v7;
        v6 = v8 - 1;
      }
      *a2 = v7;
      a2[1] = v6;
      sub_EABFE0(a1);
    }
  }
  return v2;
}
