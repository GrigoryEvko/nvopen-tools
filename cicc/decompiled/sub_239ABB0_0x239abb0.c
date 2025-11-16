// Function: sub_239ABB0
// Address: 0x239abb0
//
__int64 *__fastcall sub_239ABB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r13
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // [rsp+0h] [rbp-D0h]
  unsigned int v24; // [rsp+Ch] [rbp-C4h]
  __int64 v25; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  __int64 v27; // [rsp+18h] [rbp-B8h]
  __int64 v28; // [rsp+20h] [rbp-B0h]
  __int64 v29; // [rsp+20h] [rbp-B0h]
  __int64 v31; // [rsp+30h] [rbp-A0h]
  __int64 v32; // [rsp+38h] [rbp-98h]
  __int64 v33; // [rsp+40h] [rbp-90h] BYREF
  __int64 v34; // [rsp+48h] [rbp-88h]
  __int64 v35; // [rsp+50h] [rbp-80h]
  __int64 v36; // [rsp+58h] [rbp-78h]
  unsigned int v37; // [rsp+60h] [rbp-70h]
  __int64 v38; // [rsp+68h] [rbp-68h]
  __int64 v39; // [rsp+70h] [rbp-60h]
  __int64 v40; // [rsp+78h] [rbp-58h]
  unsigned int v41; // [rsp+80h] [rbp-50h]
  unsigned __int64 *v42; // [rsp+88h] [rbp-48h]
  unsigned __int64 *v43; // [rsp+90h] [rbp-40h]
  __int64 v44; // [rsp+98h] [rbp-38h]

  sub_11FC320((__int64)&v33, a2 + 8, a3);
  v3 = v35;
  v35 = 0;
  v24 = v37;
  ++v34;
  v4 = v41;
  v5 = (unsigned __int64)v42;
  v28 = v44;
  ++v38;
  v6 = v43;
  v23 = v33;
  v31 = v3;
  v25 = v36;
  v36 = 0;
  v37 = 0;
  v32 = v39;
  v26 = v40;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v44 = 0;
  v43 = 0;
  v42 = 0;
  v7 = sub_22077B0(0x68u);
  v8 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 16) = 1;
    *(_QWORD *)(v7 + 24) = v3;
    *(_DWORD *)(v7 + 40) = v24;
    *(_QWORD *)v7 = &unk_4A0B100;
    *(_QWORD *)(v7 + 8) = v23;
    *(_QWORD *)(v7 + 32) = v25;
    *(_QWORD *)(v7 + 48) = 1;
    *(_QWORD *)(v7 + 56) = v32;
    *(_QWORD *)(v7 + 64) = v26;
    *(_DWORD *)(v7 + 72) = v4;
    *(_QWORD *)(v7 + 80) = v5;
    *(_QWORD *)(v7 + 88) = v6;
    *(_QWORD *)(v7 + 96) = v28;
    v32 = 0;
    v31 = 0;
    v27 = 0;
    v29 = 0;
  }
  else
  {
    v29 = 16LL * v24;
    v27 = 16 * v4;
    if ( (unsigned __int64 *)v5 != v6 )
    {
      v17 = (unsigned __int64 *)v5;
      do
      {
        v18 = *v17;
        if ( *v17 )
        {
          v19 = *(_QWORD *)(v18 + 176);
          if ( v19 != v18 + 192 )
            _libc_free(v19);
          v20 = *(_QWORD *)(v18 + 88);
          if ( v20 != v18 + 104 )
            _libc_free(v20);
          v21 = 8LL * *(unsigned int *)(v18 + 80);
          sub_C7D6A0(*(_QWORD *)(v18 + 64), v21, 8);
          sub_11FC810(v18 + 32, v21);
          v22 = *(_QWORD *)(v18 + 8);
          if ( v22 != v18 + 24 )
            _libc_free(v22);
          j_j___libc_free_0(v18);
        }
        ++v17;
      }
      while ( v6 != v17 );
    }
    if ( v5 )
      j_j___libc_free_0(v5);
  }
  sub_C7D6A0(v32, v27, 8);
  sub_C7D6A0(v31, v29, 8);
  v9 = v43;
  v10 = v42;
  *a1 = v8;
  if ( v9 != v10 )
  {
    do
    {
      v11 = *v10;
      if ( *v10 )
      {
        v12 = *(_QWORD *)(v11 + 176);
        if ( v12 != v11 + 192 )
          _libc_free(v12);
        v13 = *(_QWORD *)(v11 + 88);
        if ( v13 != v11 + 104 )
          _libc_free(v13);
        v14 = 8LL * *(unsigned int *)(v11 + 80);
        sub_C7D6A0(*(_QWORD *)(v11 + 64), v14, 8);
        sub_11FC810(v11 + 32, v14);
        v15 = *(_QWORD *)(v11 + 8);
        if ( v15 != v11 + 24 )
          _libc_free(v15);
        j_j___libc_free_0(v11);
      }
      ++v10;
    }
    while ( v9 != v10 );
    v10 = v42;
  }
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  sub_C7D6A0(v39, 16LL * v41, 8);
  sub_C7D6A0(v35, 16LL * v37, 8);
  return a1;
}
