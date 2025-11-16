// Function: sub_39A4610
// Address: 0x39a4610
//
void __fastcall sub_39A4610(__int64 *a1, __int64 a2, __int64 a3, __int16 a4, __int64 a5)
{
  __int64 v6; // rax
  __int16 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rdi
  size_t v10; // rdx
  size_t v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r13
  const void *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rcx
  int v25; // r9d
  void *v26; // r8
  unsigned __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 (*v31)(); // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 *v37; // [rsp+10h] [rbp-120h]
  __int64 v38; // [rsp+18h] [rbp-118h]
  char v40; // [rsp+27h] [rbp-109h]
  void *s2; // [rsp+30h] [rbp-100h]
  void *s2a; // [rsp+30h] [rbp-100h]
  __int64 v44; // [rsp+38h] [rbp-F8h]
  unsigned __int64 *v45[2]; // [rsp+40h] [rbp-F0h] BYREF
  unsigned __int64 *v46; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-D8h]
  _QWORD v48[6]; // [rsp+60h] [rbp-D0h] BYREF
  _BYTE v49[8]; // [rsp+90h] [rbp-A0h] BYREF
  char *v50; // [rsp+98h] [rbp-98h]
  char v51; // [rsp+A8h] [rbp-88h] BYREF
  int v52; // [rsp+DCh] [rbp-54h]
  __int64 **v53; // [rsp+F8h] [rbp-38h]

  v6 = sub_3988770(a2);
  v7 = *(_WORD *)(v6 + 2);
  v8 = v6;
  v9 = *(_QWORD *)(*(_QWORD *)a2 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)a2 + 8LL)));
  s2 = (void *)v9;
  if ( v9 )
  {
    s2 = (void *)sub_161E970(v9);
    v11 = v10;
  }
  else
  {
    v11 = 0;
  }
  v40 = 0;
  if ( v7 == 15 )
  {
    v40 = 1;
    v8 = *(_QWORD *)(v8 + 8 * (3LL - *(unsigned int *)(v8 + 8)));
  }
  v12 = *(_QWORD *)(v8 + 8 * (4LL - *(unsigned int *)(v8 + 8)));
  if ( !v12 || (v13 = *(unsigned int *)(v12 + 8), !(_DWORD)v13) )
    BUG();
  v37 = a1;
  v14 = *(unsigned int *)(v12 + 8);
  v15 = 0;
  v38 = 0;
  v44 = 0;
  while ( 1 )
  {
    v16 = *(_QWORD *)(v12 + 8 * (v15 - v13));
    v17 = *(const void **)(v16 + 8 * (2LL - *(unsigned int *)(v16 + 8)));
    if ( !v17 )
      break;
    v18 = sub_161E970((__int64)v17);
    v17 = (const void *)v18;
    if ( v19 != 12 || *(_QWORD *)v18 != 0x726177726F665F5FLL || *(_DWORD *)(v18 + 8) != 1735289188 )
      goto LABEL_8;
    ++v15;
    v38 = v16;
    if ( v14 == v15 )
      goto LABEL_16;
LABEL_10:
    v13 = *(unsigned int *)(v12 + 8);
  }
  v19 = 0;
LABEL_8:
  if ( v11 == v19 )
  {
    if ( v11 )
    {
      if ( memcmp(v17, s2, v11) )
        v16 = v44;
      v44 = v16;
    }
    else
    {
      v44 = v16;
    }
  }
  if ( v14 != ++v15 )
    goto LABEL_10;
LABEL_16:
  v20 = *(_QWORD *)(v44 + 40);
  s2a = (void *)(*(_QWORD *)(v38 + 40) >> 3);
  v21 = sub_145CBF0(v37 + 11, 16, 16);
  v22 = v20 >> 2;
  *(_QWORD *)v21 = 0;
  v23 = v37[24];
  *(_DWORD *)(v21 + 8) = 0;
  sub_39A1E10((__int64)v49, v23, (__int64)v37, v21);
  v26 = s2a;
  if ( !*(_BYTE *)a5 )
    v52 = 2;
  v46 = v48;
  v47 = 0x600000000LL;
  if ( v40 )
  {
    v48[0] = 6;
    LODWORD(v47) = 1;
    if ( (_DWORD)s2a )
    {
      v33 = 1;
      goto LABEL_44;
    }
    v35 = 1;
LABEL_47:
    v27 = &v46[v35];
  }
  else
  {
    if ( !(_DWORD)s2a )
    {
      v27 = v48;
      goto LABEL_21;
    }
    v33 = 0;
LABEL_44:
    v48[v33] = 35;
    v34 = (unsigned int)(v47 + 1);
    LODWORD(v47) = v34;
    if ( HIDWORD(v47) <= (unsigned int)v34 )
    {
      sub_16CD150((__int64)&v46, v48, 0, 8, (int)s2a, v25);
      v34 = (unsigned int)v47;
    }
    v46[v34] = (unsigned int)s2a;
    v35 = (unsigned int)(v47 + 1);
    LODWORD(v47) = v35;
    if ( HIDWORD(v47) > (unsigned int)v35 )
      goto LABEL_47;
    sub_16CD150((__int64)&v46, v48, 0, 8, (int)v26, v25);
    v27 = &v46[(unsigned int)v47];
  }
LABEL_21:
  *v27 = 6;
  v28 = (unsigned int)(v47 + 1);
  LODWORD(v47) = v47 + 1;
  if ( (_DWORD)v22 )
  {
    if ( (unsigned int)v28 >= HIDWORD(v47) )
    {
      sub_16CD150((__int64)&v46, v48, 0, 8, (int)v26, v25);
      v28 = (unsigned int)v47;
    }
    v46[v28] = 35;
    v32 = (unsigned int)(v47 + 1);
    LODWORD(v47) = v32;
    if ( HIDWORD(v47) <= (unsigned int)v32 )
    {
      sub_16CD150((__int64)&v46, v48, 0, 8, (int)v26, v25);
      v32 = (unsigned int)v47;
    }
    v46[v32] = (unsigned int)v22;
    v28 = (unsigned int)(v47 + 1);
    LODWORD(v47) = v47 + 1;
  }
  v29 = 0;
  v45[0] = v46;
  v45[1] = &v46[v28];
  v30 = *(_QWORD *)(*(_QWORD *)(v37[24] + 264) + 16LL);
  v31 = *(__int64 (**)())(*(_QWORD *)v30 + 112LL);
  if ( v31 != sub_1D00B10 )
    v29 = ((__int64 (__fastcall *)(__int64, _QWORD, unsigned __int64 *, __int64, void *))v31)(v30, 0, v46, v24, v26);
  if ( (unsigned __int8)sub_399F750((__int64)v49, v29, v45, *(_DWORD *)(a5 + 4)) )
  {
    sub_399FAC0(v49, v45, 0);
    sub_399FD30((__int64)v49);
    sub_39A4520(v37, a3, a4, v53);
  }
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( v50 != &v51 )
    _libc_free((unsigned __int64)v50);
}
