// Function: sub_9A74C0
// Address: 0x9a74c0
//
__int64 __fastcall sub_9A74C0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 **a3,
        __int64 a4,
        int a5,
        const __m128i *a6,
        void (__fastcall *a7)(unsigned __int64 **, __int64, __int64, unsigned __int64 **, _QWORD),
        __int64 a8)
{
  unsigned int v10; // r12d
  _QWORD *v12; // rdi
  __int64 v13; // rdx
  int v14; // r8d
  unsigned int v15; // eax
  bool v16; // cc
  unsigned int v17; // eax
  unsigned __int64 *v18; // rdi
  __int64 result; // rax
  unsigned int v20; // r8d
  unsigned __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  __int64 v25; // r11
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // eax
  unsigned __int64 v29; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v30; // [rsp+10h] [rbp-90h]
  int v31; // [rsp+20h] [rbp-80h]
  unsigned int v32; // [rsp+20h] [rbp-80h]
  unsigned int v33; // [rsp+20h] [rbp-80h]
  unsigned int v34; // [rsp+20h] [rbp-80h]
  unsigned __int64 v36; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-58h]
  unsigned __int64 *v38; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v40; // [rsp+60h] [rbp-40h]
  unsigned int v41; // [rsp+68h] [rbp-38h]

  v10 = a5 + 1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v12 = *(_QWORD **)(a1 - 8);
  else
    v12 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  sub_9AB8E0(*v12, a2, a4, v10, a6);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v13 = *(_QWORD *)(a1 - 8);
  else
    v13 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  sub_9AB8E0(*(_QWORD *)(v13 + 32), a2, a3, v10, a6);
  if ( *((_DWORD *)a3 + 6) <= 0x40u )
  {
    if ( a3[2] )
    {
      v15 = 1;
      goto LABEL_7;
    }
  }
  else
  {
    v31 = *((_DWORD *)a3 + 6);
    v14 = sub_C444A0(a3 + 2);
    v15 = 1;
    if ( v31 != v14 )
      goto LABEL_7;
  }
  v20 = *((_DWORD *)a3 + 2);
  v39 = v20;
  if ( v20 <= 0x40 )
  {
    v21 = (unsigned __int64)*a3;
    v22 = v20;
LABEL_19:
    v37 = v20;
    v23 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & ~v21;
    if ( !v20 )
      v23 = 0;
    v24 = v22;
    v36 = v23;
LABEL_22:
    v15 = 0;
    if ( v24 <= v23 )
      goto LABEL_7;
LABEL_23:
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      v25 = *(_QWORD *)(a1 - 8);
    else
      v25 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    v32 = v20;
    v15 = (unsigned __int8)sub_9A6530(*(_QWORD *)(v25 + 32), a2, a6, v10);
    if ( v32 <= 0x40 )
      goto LABEL_7;
    goto LABEL_26;
  }
  sub_C43780(&v38, a3);
  v20 = v39;
  if ( v39 <= 0x40 )
  {
    v21 = (unsigned __int64)v38;
    v22 = *((_DWORD *)a3 + 2);
    goto LABEL_19;
  }
  sub_C43D10(&v38, a3, v26, v27, v39);
  v20 = v39;
  v23 = (unsigned __int64)v38;
  v24 = *((unsigned int *)a3 + 2);
  v37 = v39;
  v36 = (unsigned __int64)v38;
  if ( v39 <= 0x40 )
    goto LABEL_22;
  v34 = v39;
  v29 = v24;
  v30 = v38;
  v28 = sub_C444A0(&v36);
  v20 = v34;
  if ( v34 - v28 <= 0x40 && v29 > *v30 )
    goto LABEL_23;
  v15 = 0;
LABEL_26:
  if ( v36 )
  {
    v33 = v15;
    j_j___libc_free_0_0(v36);
    v15 = v33;
  }
LABEL_7:
  a7(&v38, a8, a4, a3, v15);
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    j_j___libc_free_0_0(*a3);
  v16 = *((_DWORD *)a3 + 6) <= 0x40u;
  *a3 = v38;
  v17 = v39;
  v39 = 0;
  *((_DWORD *)a3 + 2) = v17;
  if ( v16 || (v18 = a3[2]) == 0 )
  {
    a3[2] = v40;
    result = v41;
    *((_DWORD *)a3 + 6) = v41;
  }
  else
  {
    j_j___libc_free_0_0(v18);
    v16 = v39 <= 0x40;
    a3[2] = v40;
    result = v41;
    *((_DWORD *)a3 + 6) = v41;
    if ( !v16 )
    {
      if ( v38 )
        return j_j___libc_free_0_0(v38);
    }
  }
  return result;
}
