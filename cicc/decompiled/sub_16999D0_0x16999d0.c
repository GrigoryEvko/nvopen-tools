// Function: sub_16999D0
// Address: 0x16999d0
//
__int64 __fastcall sub_16999D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r10
  __int64 *p_src; // r12
  int v5; // r15d
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // eax
  unsigned int v10; // r14d
  __int64 *v11; // r10
  __int16 v12; // cx
  unsigned int v13; // eax
  __int16 v14; // dx
  __int64 *v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  __int64 v19; // rax
  unsigned int v20; // r15d
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int16 v23; // dx
  __int64 v24; // rax
  int v25; // eax
  __int64 *v26; // [rsp+0h] [rbp-B0h]
  __int64 v27; // [rsp+8h] [rbp-A8h]
  __int16 v28; // [rsp+10h] [rbp-A0h]
  bool v29; // [rsp+17h] [rbp-99h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 *v31; // [rsp+18h] [rbp-98h]
  unsigned int v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+20h] [rbp-90h]
  unsigned int v35; // [rsp+28h] [rbp-88h]
  unsigned int v36; // [rsp+28h] [rbp-88h]
  unsigned int v37; // [rsp+2Ch] [rbp-84h]
  bool v38; // [rsp+33h] [rbp-7Dh] BYREF
  __int64 v39; // [rsp+34h] [rbp-7Ch] BYREF
  int v40; // [rsp+3Ch] [rbp-74h]
  _QWORD v41[4]; // [rsp+40h] [rbp-70h] BYREF
  char src; // [rsp+60h] [rbp-50h] BYREF

  v3 = a3;
  p_src = (__int64 *)&src;
  v29 = 1;
  v5 = 2 * *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v35 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v37 = (unsigned int)(v5 + 64) >> 6;
  if ( (unsigned int)(v5 + 64) > 0x13F )
  {
    v19 = sub_2207820(8LL * ((unsigned int)(v5 + 64) >> 6));
    v3 = a3;
    p_src = (__int64 *)v19;
    v29 = v19 != 0;
  }
  v30 = v3;
  v33 = sub_1698470(a1);
  v6 = sub_1698310(a1);
  v7 = sub_16984A0(a2);
  sub_16A7C60(p_src, v33, v7, v6, v6);
  v9 = sub_16A7150(p_src, v37, v8);
  v10 = 0;
  v11 = (__int64 *)v30;
  v12 = v9;
  v13 = v9 + 1;
  v14 = *(_WORD *)(a2 + 16) + *(_WORD *)(a1 + 16) + 2;
  *(_WORD *)(a1 + 16) = v14;
  if ( v30 && (*(_BYTE *)(v30 + 18) & 7) != 3 )
  {
    v27 = *(_QWORD *)(a1 + 8);
    v31 = *(__int64 **)a1;
    v15 = *(__int64 **)a1;
    if ( v5 != v13 )
    {
      v26 = v11;
      v28 = v12;
      sub_16A7D00(p_src);
      v15 = *(__int64 **)a1;
      v11 = v26;
      *(_WORD *)(a1 + 16) = *(_WORD *)(a1 + 16) + 1 - v5 + v28;
    }
    v39 = *v15;
    v16 = *((_DWORD *)v15 + 2);
    HIDWORD(v39) = v5 + 1;
    v40 = v16;
    if ( v37 == 1 )
    {
      v24 = *p_src;
      *(_QWORD *)a1 = &v39;
      *(_QWORD *)(a1 + 8) = v24;
      sub_16986C0(v41, v11);
      sub_16995F0((__int64)v41, (__int16 *)&v39, 3u, &v38);
      sub_1698C00((__int64)v41, 1u);
      v10 = sub_16991E0(a1, (__int64)v41, 0);
      *p_src = *(_QWORD *)(a1 + 8);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = p_src;
      *(_QWORD *)a1 = &v39;
      sub_16986C0(v41, v11);
      sub_16995F0((__int64)v41, (__int16 *)&v39, 3u, &v38);
      sub_1698C00((__int64)v41, 1u);
      v10 = sub_16991E0(a1, (__int64)v41, 0);
    }
    *(_QWORD *)(a1 + 8) = v27;
    *(_QWORD *)a1 = v31;
    v32 = sub_16A7150(p_src, v37, v17) + 1;
    sub_1698460((__int64)v41);
    v14 = *(_WORD *)(a1 + 16);
    v13 = v32;
  }
  *(_WORD *)(a1 + 16) = ~(_WORD)v35 + v14;
  if ( v13 > v35 )
  {
    v20 = (v13 + 63) >> 6;
    v36 = v13 - v35;
    v21 = sub_16A7110(p_src, v20);
    v22 = v36;
    if ( v36 <= v21 )
    {
      sub_16A8050(p_src, v20, v36);
      v23 = v36;
      v10 = v10 != 0;
    }
    else if ( v36 == v21 + 1 )
    {
      sub_16A8050(p_src, v20, v36);
      v23 = v36;
      v10 = 2 - ((v10 == 0) - 1);
    }
    else if ( v36 <= v20 << 6 && (v25 = sub_16A70B0(p_src, v36 - 1), v22 = v36, v25) )
    {
      v10 = 3;
      sub_16A8050(p_src, v20, v36);
      v23 = v36;
    }
    else
    {
      v10 = 1;
      sub_16A8050(p_src, v20, v22);
      v23 = v36;
    }
    *(_WORD *)(a1 + 16) += v23;
  }
  sub_16A7050(v33, p_src, v6);
  if ( v37 > 4 && v29 )
    j_j___libc_free_0_0(p_src);
  return v10;
}
