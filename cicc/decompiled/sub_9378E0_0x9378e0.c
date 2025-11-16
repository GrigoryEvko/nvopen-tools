// Function: sub_9378E0
// Address: 0x9378e0
//
__int64 __fastcall sub_9378E0(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 **a4,
        unsigned __int8 **a5,
        unsigned __int8 **a6)
{
  unsigned __int8 *v6; // r14
  unsigned __int8 *v7; // rax
  unsigned __int64 *v8; // r12
  unsigned __int8 *v9; // r13
  __int64 v10; // rdx
  unsigned __int8 *v11; // rbx
  unsigned __int8 *v12; // rdi
  unsigned __int64 v13; // rbx
  _QWORD *v14; // rcx
  _QWORD *v15; // r11
  unsigned __int64 *v16; // r15
  __int64 v17; // rdx
  unsigned __int8 *v18; // r12
  int v19; // r8d
  __int64 v20; // rdx
  int v21; // ebx
  __int64 v22; // rdx
  int v23; // ebx
  __int64 v24; // rdx
  int v25; // ebx
  unsigned __int64 v26; // rbx
  __int64 v27; // rdx
  _QWORD *v28; // rsi
  __int64 v29; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  _QWORD *v38; // [rsp+38h] [rbp-108h]
  _QWORD *v39; // [rsp+38h] [rbp-108h]
  _QWORD *v40; // [rsp+38h] [rbp-108h]
  _QWORD *v41; // [rsp+38h] [rbp-108h]
  _QWORD *v42; // [rsp+38h] [rbp-108h]
  unsigned __int64 *v44; // [rsp+50h] [rbp-F0h]
  unsigned __int8 *v45; // [rsp+58h] [rbp-E8h]
  unsigned __int8 *v46; // [rsp+60h] [rbp-E0h]
  unsigned __int8 *v47; // [rsp+68h] [rbp-D8h]
  __int64 v48; // [rsp+78h] [rbp-C8h] BYREF
  _QWORD *v49; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+88h] [rbp-B8h]
  _QWORD v51[22]; // [rsp+90h] [rbp-B0h] BYREF

  v6 = *a5;
  v7 = *a6;
  v8 = *(unsigned __int64 **)a3;
  v9 = *a4;
  v10 = *(unsigned int *)(a3 + 8);
  v11 = &a6[1][(_QWORD)*a6];
  v46 = &a5[1][(_QWORD)*a5];
  v12 = a4[1];
  v51[0] = a2;
  v49 = v51;
  v45 = v11;
  v47 = &v12[(_QWORD)v9];
  v50 = 0x2000000002LL;
  if ( &v8[v10] == v8 || v9 == v47 || v6 == v46 || v7 == v11 )
  {
    v15 = &v49;
  }
  else
  {
    v13 = *v8;
    v14 = v51;
    v15 = &v49;
    v16 = v8;
    v44 = &v8[v10];
    v17 = 2;
    v18 = v7;
    v19 = v13;
    while ( 1 )
    {
      *((_DWORD *)v14 + v17) = v19;
      v26 = HIDWORD(v13);
      LODWORD(v50) = v50 + 1;
      v27 = (unsigned int)v50;
      if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
      {
        v38 = v15;
        sub_C8D5F0(v15, v51, (unsigned int)v50 + 1LL, 4);
        v27 = (unsigned int)v50;
        v15 = v38;
      }
      *((_DWORD *)v49 + v27) = v26;
      LODWORD(v50) = v50 + 1;
      v20 = (unsigned int)v50;
      v21 = *v9;
      if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
      {
        v42 = v15;
        sub_C8D5F0(v15, v51, (unsigned int)v50 + 1LL, 4);
        v20 = (unsigned int)v50;
        v15 = v42;
      }
      *((_DWORD *)v49 + v20) = v21;
      LODWORD(v50) = v50 + 1;
      v22 = (unsigned int)v50;
      v23 = *v6;
      if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
      {
        v41 = v15;
        sub_C8D5F0(v15, v51, (unsigned int)v50 + 1LL, 4);
        v22 = (unsigned int)v50;
        v15 = v41;
      }
      *((_DWORD *)v49 + v22) = v23;
      LODWORD(v50) = v50 + 1;
      v24 = (unsigned int)v50;
      v25 = *v18;
      if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
      {
        v40 = v15;
        sub_C8D5F0(v15, v51, (unsigned int)v50 + 1LL, 4);
        v24 = (unsigned int)v50;
        v15 = v40;
      }
      ++v16;
      ++v9;
      ++v6;
      ++v18;
      *((_DWORD *)v49 + v24) = v25;
      v17 = (unsigned int)(v50 + 1);
      LODWORD(v50) = v50 + 1;
      if ( v44 == v16 || v47 == v9 || v46 == v6 || v45 == v18 )
        break;
      v13 = *v16;
      v19 = *v16;
      if ( v17 + 1 > (unsigned __int64)HIDWORD(v50) )
      {
        v39 = v15;
        sub_C8D5F0(v15, v51, v17 + 1, 4);
        v17 = (unsigned int)v50;
        v19 = v13;
        v15 = v39;
      }
      v14 = v49;
    }
  }
  v28 = v15;
  v48 = 0;
  v29 = sub_C65B40(a1 + 17, v15, &v48, off_49D9480);
  if ( !v29 )
  {
    v31 = sub_22077B0(24);
    v29 = v31;
    if ( v31 )
      sub_9377A0(v31, a2, a3, a4, a5, a6);
    sub_C657C0(a1 + 17, v29, v48, off_49D9480);
    v32 = sub_947280(a1);
    v28 = (_QWORD *)v29;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD **))(*(_QWORD *)v32 + 16LL))(v32, v29, **a1, a1);
  }
  if ( v49 != v51 )
    _libc_free(v49, v28);
  return v29;
}
