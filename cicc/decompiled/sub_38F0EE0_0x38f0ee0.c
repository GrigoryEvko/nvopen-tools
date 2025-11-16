// Function: sub_38F0EE0
// Address: 0x38f0ee0
//
__int64 __fastcall sub_38F0EE0(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4)
{
  int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  _DWORD *v17; // rsi
  _DWORD *v18; // r13
  int v19; // edx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rax
  __m128i v22; // xmm0
  bool v23; // cc
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rdx
  _DWORD *v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  bool v30; // zf
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // [rsp+0h] [rbp-A0h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  int v35; // [rsp+10h] [rbp-90h] BYREF
  __int64 v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+20h] [rbp-80h]
  unsigned __int64 v38; // [rsp+28h] [rbp-78h]
  unsigned int v39; // [rsp+30h] [rbp-70h]
  _BYTE v40[24]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v41; // [rsp+58h] [rbp-48h]
  unsigned int v42; // [rsp+60h] [rbp-40h]

  v6 = **(_DWORD **)(a1 + 152);
  LOBYTE(a4) = v6 == 26 || v6 == 45;
  v7 = a4;
  if ( (_BYTE)a4 )
  {
    v15 = a1 + 144;
    v34 = sub_3909290(a1 + 144);
    v36 = 0;
    v37 = 0;
    v39 = 1;
    v38 = 0;
    sub_392A3E0(a1 + 144, &v35, 1, 0);
    if ( v35 == 2 && v34 + 1 == sub_39092A0(&v35) )
    {
      v16 = *(unsigned int *)(a1 + 160);
      v17 = *(_DWORD **)(a1 + 152);
      v18 = v17 + 10;
      v19 = *(_DWORD *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = *v17 == 9;
      v20 = 40 * v16;
      v21 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v16 - 40) >> 3);
      if ( v20 > 0x28 )
      {
        do
        {
          v22 = _mm_loadu_si128((const __m128i *)(v18 + 2));
          v23 = *(v18 - 2) <= 0x40u;
          *(v18 - 10) = *v18;
          *((__m128i *)v18 - 2) = v22;
          if ( !v23 )
          {
            v24 = *((_QWORD *)v18 - 2);
            if ( v24 )
            {
              v33 = v21;
              j_j___libc_free_0_0(v24);
              v21 = v33;
            }
          }
          v25 = *((_QWORD *)v18 + 3);
          v18 += 10;
          *((_QWORD *)v18 - 7) = v25;
          LODWORD(v25) = *(v18 - 2);
          *(v18 - 2) = 0;
          *(v18 - 12) = v25;
          --v21;
        }
        while ( v21 );
        v19 = *(_DWORD *)(a1 + 160);
        v17 = *(_DWORD **)(a1 + 152);
      }
      v26 = (unsigned int)(v19 - 1);
      *(_DWORD *)(a1 + 160) = v26;
      v27 = &v17[10 * v26];
      if ( v27[8] > 0x40u )
      {
        v28 = *((_QWORD *)v27 + 3);
        if ( v28 )
          j_j___libc_free_0_0(v28);
      }
      if ( !*(_DWORD *)(a1 + 160) )
      {
        sub_392C2E0(v40, v15);
        sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)v40);
        if ( v42 > 0x40 )
        {
          if ( v41 )
            j_j___libc_free_0_0(v41);
        }
      }
      v29 = sub_3909460(a1);
      v30 = *(_DWORD *)v29 == 2;
      v31 = *(_QWORD *)(v29 + 16);
      if ( !v30 && v31 )
      {
        v32 = v31 - 1;
        if ( v31 == 1 )
          v32 = 1;
        if ( v32 <= v31 )
          v31 = v32;
        --v31;
      }
      v7 = 0;
      a2[1] = v31 + 1;
      *a2 = v34;
      sub_38EB180(a1);
    }
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
  }
  else if ( (unsigned int)(v6 - 2) > 1 )
  {
    return 1;
  }
  else
  {
    v8 = sub_3909460(a1);
    v9 = v8;
    if ( *(_DWORD *)v8 == 2 )
    {
      v13 = *(_QWORD *)(v8 + 8);
      v11 = *(_QWORD *)(v9 + 16);
    }
    else
    {
      v10 = *(_QWORD *)(v8 + 16);
      v11 = 0;
      if ( v10 )
      {
        v12 = v10 - 1;
        if ( v10 == 1 )
          v12 = 1;
        if ( v12 > v10 )
          v12 = v10;
        v10 = 1;
        v11 = v12 - 1;
      }
      v13 = *(_QWORD *)(v9 + 8) + v10;
    }
    *a2 = v13;
    a2[1] = v11;
    sub_38EB180(a1);
  }
  return v7;
}
