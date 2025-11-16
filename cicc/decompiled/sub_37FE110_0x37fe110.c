// Function: sub_37FE110
// Address: 0x37fe110
//
void __fastcall sub_37FE110(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v7; // r9
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  _DWORD *v13; // rbx
  __int64 *v14; // rsi
  __int64 v15; // rsi
  int v16; // edx
  __int64 v17; // r15
  int v18; // edx
  _QWORD *i; // rbx
  __int64 v20; // rdx
  _QWORD *v21; // rdi
  _QWORD *v22; // r12
  __int64 v23; // rdx
  __int64 v26; // [rsp+18h] [rbp-C8h]
  _DWORD *v27; // [rsp+28h] [rbp-B8h]
  unsigned int v28; // [rsp+50h] [rbp-90h] BYREF
  __int64 v29; // [rsp+58h] [rbp-88h]
  unsigned __int64 v30; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v31; // [rsp+68h] [rbp-78h]
  __int64 v32; // [rsp+70h] [rbp-70h] BYREF
  int v33; // [rsp+78h] [rbp-68h]
  unsigned __int64 v34; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+88h] [rbp-58h]
  _DWORD *v36; // [rsp+90h] [rbp-50h] BYREF
  _QWORD *v37; // [rsp+98h] [rbp-48h]
  __int64 v38; // [rsp+A0h] [rbp-40h]

  v7 = *a1;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v36, v7, *(_QWORD *)(v12 + 64), v10, v11);
    LOWORD(v28) = (_WORD)v37;
    v29 = v38;
  }
  else
  {
    v28 = v8(v7, *(_QWORD *)(v12 + 64), v10, v11);
    v29 = v23;
  }
  v13 = sub_C33340();
  v14 = (__int64 *)(*(_QWORD *)(a2 + 96) + 24LL);
  if ( (_DWORD *)*v14 == v13 )
    sub_C3E660((__int64)&v30, (__int64)v14);
  else
    sub_C3A850((__int64)&v30, v14);
  v15 = *(_QWORD *)(a2 + 80);
  v32 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v32, v15, 1);
  v33 = *(_DWORD *)(a2 + 72);
  v27 = sub_300AC80((unsigned __int16 *)&v28, v15);
  v26 = a1[1];
  sub_C440A0((__int64)&v34, (__int64 *)&v30, 0x40u, 0x40u);
  if ( v27 == v13 )
    sub_C3C640(&v36, (__int64)v13, &v34);
  else
    sub_C3B160((__int64)&v36, v27, (__int64 *)&v34);
  *(_QWORD *)a3 = sub_33FE6E0(v26, (__int64 *)&v36, (__int64)&v32, v28, v29, 0, a5);
  *(_DWORD *)(a3 + 8) = v16;
  if ( v13 == v36 )
  {
    if ( v37 )
    {
      v20 = *(v37 - 1);
      v21 = &v37[3 * v20];
      if ( v37 != v21 )
      {
        v22 = &v37[3 * v20];
        do
        {
          v22 -= 3;
          sub_91D830(v22);
        }
        while ( v37 != v22 );
        v21 = v22;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v21 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v36);
  }
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  v17 = a1[1];
  sub_C440A0((__int64)&v34, (__int64 *)&v30, 0x40u, 0);
  if ( v27 == v13 )
    sub_C3C640(&v36, (__int64)v13, &v34);
  else
    sub_C3B160((__int64)&v36, v27, (__int64 *)&v34);
  *(_QWORD *)a4 = sub_33FE6E0(v17, (__int64 *)&v36, (__int64)&v32, v28, v29, 0, a5);
  *(_DWORD *)(a4 + 8) = v18;
  if ( v13 == v36 )
  {
    if ( v37 )
    {
      for ( i = &v37[3 * *(v37 - 1)]; v37 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v36);
  }
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  if ( v31 > 0x40 )
  {
    if ( v30 )
      j_j___libc_free_0_0(v30);
  }
}
