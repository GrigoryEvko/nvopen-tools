// Function: sub_3445A50
// Address: 0x3445a50
//
__int64 __fastcall sub_3445A50(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v9; // rcx
  unsigned int v10; // ebx
  __int64 v11; // rsi
  __int64 *v12; // rdi
  __int64 v13; // rax
  __int16 v14; // dx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 (__fastcall *v17)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // rdx
  unsigned int v21; // ebx
  __int64 v22; // rdx
  void *v23; // rsi
  __int128 v24; // rax
  int v25; // r9d
  unsigned __int8 *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // r15
  __int128 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r14
  __int128 v33; // rax
  _QWORD *i; // rbx
  __int128 v35; // [rsp-40h] [rbp-E0h]
  __int128 v36; // [rsp-30h] [rbp-D0h]
  __int128 v37; // [rsp+0h] [rbp-A0h]
  __int128 v38; // [rsp+0h] [rbp-A0h]
  __int64 v39; // [rsp+18h] [rbp-88h]
  unsigned int v40; // [rsp+18h] [rbp-88h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  void *v44; // [rsp+28h] [rbp-78h]
  __int64 v45; // [rsp+30h] [rbp-70h] BYREF
  int v46; // [rsp+38h] [rbp-68h]
  unsigned int v47; // [rsp+40h] [rbp-60h] BYREF
  __int64 v48; // [rsp+48h] [rbp-58h]
  void *v49; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v50; // [rsp+58h] [rbp-48h]

  v5 = a1;
  v9 = a2;
  v10 = a3;
  v11 = *(_QWORD *)(a2 + 80);
  v45 = v11;
  if ( v11 )
  {
    sub_B96E90((__int64)&v45, v11, 1);
    v5 = a1;
    v9 = a2;
  }
  v12 = (__int64 *)a4[5];
  v39 = v5;
  v46 = *(_DWORD *)(v9 + 72);
  v13 = *(_QWORD *)(v9 + 48) + 16LL * v10;
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  LOWORD(v47) = v14;
  v16 = a4[8];
  v48 = v15;
  v41 = v16;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v5 + 528LL);
  v18 = sub_2E79000(v12);
  v19 = v17(v39, v18, v41, v47, v48);
  v42 = v20;
  v21 = v19;
  v40 = v19;
  *(_QWORD *)&v37 = sub_33FE730((__int64)a4, (__int64)&v45, v47, v48, 0, (__m128i)0LL);
  *((_QWORD *)&v37 + 1) = v22;
  if ( (unsigned __int8)(*(_BYTE *)(a5 + 1) - 1) <= 1u )
  {
    *(_QWORD *)&v33 = sub_33ED040(a4, 0x11u);
    *((_QWORD *)&v36 + 1) = a3;
    *(_QWORD *)&v36 = a2;
    v31 = sub_340F900(a4, 0xD0u, (__int64)&v45, v21, v42, *((__int64 *)&v37 + 1), v36, v37, v33);
    goto LABEL_9;
  }
  v23 = sub_300AC80((unsigned __int16 *)&v47, (__int64)&v45);
  v44 = sub_C33340();
  if ( v23 == v44 )
  {
    sub_C3C500(&v49, (__int64)v44);
    if ( v49 != v44 )
      goto LABEL_6;
  }
  else
  {
    sub_C373C0(&v49, (__int64)v23);
    if ( v49 != v44 )
    {
LABEL_6:
      sub_C35A40((__int64)&v49, 0);
      goto LABEL_7;
    }
  }
  sub_C3D240((__int64)&v49, 0);
LABEL_7:
  *(_QWORD *)&v24 = sub_33FE6E0((__int64)a4, (__int64 *)&v49, (__int64)&v45, v47, v48, 0, (__m128i)0LL);
  v38 = v24;
  v26 = sub_33FAF80((__int64)a4, 245, (__int64)&v45, v47, v48, v25, (__m128i)0LL);
  v28 = v27;
  *(_QWORD *)&v29 = sub_33ED040(a4, 0x14u);
  *((_QWORD *)&v35 + 1) = v28;
  *(_QWORD *)&v35 = v26;
  v31 = sub_340F900(a4, 0xD0u, (__int64)&v45, v40, v42, v30, v35, v38, v29);
  if ( v49 == v44 )
  {
    if ( v50 )
    {
      for ( i = &v50[3 * *(v50 - 1)]; v50 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v49);
  }
LABEL_9:
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  return v31;
}
