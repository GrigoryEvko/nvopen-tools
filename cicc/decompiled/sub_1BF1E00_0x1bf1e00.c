// Function: sub_1BF1E00
// Address: 0x1bf1e00
//
void __fastcall sub_1BF1E00(
        __int64 a1,
        const char **a2,
        __int64 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  const char **v12; // r14
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  __int64 v23; // r15
  const char *v24; // rax
  unsigned int v25; // r15d
  int v26; // r8d
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 *v29; // rax
  unsigned __int8 *v30; // r12
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // [rsp-E8h] [rbp-E8h]
  __int64 v34; // [rsp-C8h] [rbp-C8h]
  __int64 v35; // [rsp-C0h] [rbp-C0h]
  const char **v36; // [rsp-C0h] [rbp-C0h]
  _QWORD v37[2]; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v38[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v39; // [rsp-98h] [rbp-98h]
  void *v40[2]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v41; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v42; // [rsp-68h] [rbp-68h] BYREF
  __int64 v43; // [rsp-60h] [rbp-60h]
  _QWORD v44[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( a3 )
  {
    v12 = a2;
    v13 = a1;
    v14 = *(_QWORD *)(a1 + 72);
    v42 = v44;
    v43 = 0x400000001LL;
    v44[0] = 0;
    v15 = sub_13FD000(v14);
    v16 = v15;
    if ( v15 )
    {
      v17 = *(unsigned int *)(v15 + 8);
      if ( (unsigned int)v17 > 1 )
      {
        v34 = v17;
        v18 = 1;
        v35 = v13;
        while ( 1 )
        {
          v19 = *(_QWORD *)(v16 + 8 * (v18 - v17));
          if ( (unsigned __int8)sub_1BF1D50(v35, v19, a2, a3) )
          {
            if ( v34 == ++v18 )
              goto LABEL_11;
          }
          else
          {
            v22 = (unsigned int)v43;
            if ( (unsigned int)v43 >= HIDWORD(v43) )
            {
              sub_16CD150((__int64)&v42, v44, 0, 8, v20, v21);
              v22 = (unsigned int)v43;
            }
            ++v18;
            v42[v22] = v19;
            LODWORD(v43) = v43 + 1;
            if ( v34 == v18 )
            {
LABEL_11:
              v13 = v35;
              break;
            }
          }
          v17 = *(unsigned int *)(v16 + 8);
        }
      }
    }
    v23 = 2 * a3;
    v36 = &a2[v23];
    if ( a2 != &a2[v23] )
    {
      do
      {
        v24 = *v12;
        v37[0] = "llvm.loop.";
        v25 = *((_DWORD *)v12 + 2);
        v39 = 773;
        v38[0] = v37;
        v38[1] = v24;
        v37[1] = 10;
        sub_16E2FC0((__int64 *)v40, (__int64)v38);
        v27 = sub_1BF1CC0(v13, v40[0], (size_t)v40[1], v25);
        v28 = (unsigned int)v43;
        if ( (unsigned int)v43 >= HIDWORD(v43) )
        {
          v33 = v27;
          sub_16CD150((__int64)&v42, v44, 0, 8, v26, v27);
          v28 = (unsigned int)v43;
          v27 = v33;
        }
        v42[v28] = v27;
        LODWORD(v43) = v43 + 1;
        if ( v40[0] != &v41 )
          j_j___libc_free_0(v40[0], v41 + 1);
        v12 += 2;
      }
      while ( v36 != v12 );
    }
    v29 = (__int64 *)sub_157E9C0(**(_QWORD **)(*(_QWORD *)(v13 + 72) + 32LL));
    v30 = (unsigned __int8 *)sub_1627350(v29, v42, (__int64 *)(unsigned int)v43, 0, 1);
    sub_1630830((__int64)v30, 0, v30, a4, a5, a6, a7, v31, v32, a10, a11);
    sub_13FCC30(*(_QWORD *)(v13 + 72), (__int64)v30);
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
  }
}
