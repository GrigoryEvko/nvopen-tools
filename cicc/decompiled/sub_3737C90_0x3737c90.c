// Function: sub_3737C90
// Address: 0x3737c90
//
void __fastcall sub_3737C90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned __int64 **v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  const void *v9; // r11
  __int64 v10; // r9
  __int64 v11; // rdx
  size_t v12; // rax
  __int64 v13; // r8
  unsigned __int64 *v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 (*v17)(void); // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 (*v20)(); // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // eax
  unsigned int v30; // [rsp+0h] [rbp-180h]
  char v31; // [rsp+7h] [rbp-179h]
  __int64 v33; // [rsp+10h] [rbp-170h]
  size_t v34; // [rsp+18h] [rbp-168h]
  __int64 v35; // [rsp+20h] [rbp-160h]
  int v36; // [rsp+28h] [rbp-158h]
  __int64 v37; // [rsp+28h] [rbp-158h]
  const void *v38; // [rsp+28h] [rbp-158h]
  __int64 i; // [rsp+38h] [rbp-148h]
  unsigned int v40; // [rsp+4Ch] [rbp-134h] BYREF
  _QWORD v41[2]; // [rsp+50h] [rbp-130h] BYREF
  unsigned __int64 *v42[2]; // [rsp+60h] [rbp-120h] BYREF
  unsigned __int64 *v43; // [rsp+70h] [rbp-110h] BYREF
  __int64 v44; // [rsp+78h] [rbp-108h]
  _BYTE v45[64]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD v46[3]; // [rsp+C0h] [rbp-C0h] BYREF
  char *v47; // [rsp+D8h] [rbp-A8h]
  char v48; // [rsp+E8h] [rbp-98h] BYREF
  char v49; // [rsp+124h] [rbp-5Ch]
  unsigned __int8 v50; // [rsp+126h] [rbp-5Ah]
  char v51; // [rsp+127h] [rbp-59h]
  __int64 **v52; // [rsp+130h] [rbp-50h]

  v5 = sub_A777F0(0x10u, a1 + 11);
  v6 = (unsigned __int64 **)v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_DWORD *)(v5 + 8) = 0;
  }
  sub_3247620((__int64)v46, a1[23], (__int64)a1, v5);
  v7 = sub_321E240(a2);
  v31 = 0;
  v8 = *(_QWORD *)(v7 + 24);
  v30 = 0;
  for ( i = v7 + 8; i != v8; v8 = sub_220EF30(v8) )
  {
    v19 = a1[23];
    v40 = 0;
    v20 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(v19 + 232) + 16LL) + 136LL);
    if ( v20 == sub_2DD19D0 )
      BUG();
    v37 = *(_QWORD *)(v8 + 40);
    v21 = v20();
    v22 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, unsigned int *))(*(_QWORD *)v21 + 224LL))(
            v21,
            *(_QWORD *)(a1[23] + 232),
            *(unsigned int *)(v8 + 32),
            &v40);
    v41[1] = v23;
    v41[0] = v22;
    sub_3243D60(v46, v37);
    v24 = *(_QWORD *)(*(_QWORD *)(a1[23] + 232) + 16LL);
    v25 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 200LL))(v24);
    v43 = (unsigned __int64 *)v45;
    v44 = 0x800000000LL;
    (*(void (__fastcall **)(__int64, _QWORD *, unsigned __int64 **))(*(_QWORD *)v25 + 592LL))(v25, v41, &v43);
    v26 = v37;
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1 && *(_DWORD *)(a1[26] + 6224) == 1 )
    {
      v28 = sub_B0D5A0(v37, v42);
      if ( v37 != v28 )
      {
        v31 = 1;
        v30 = (unsigned int)v42[0];
      }
      v26 = v28;
    }
    v15 = (unsigned int)v44;
    if ( v26 )
    {
      v9 = *(const void **)(v26 + 16);
      v10 = *(_QWORD *)(v26 + 24);
      v11 = (unsigned int)v44;
      v12 = v10 - (_QWORD)v9;
      v13 = (v10 - (__int64)v9) >> 3;
      if ( v13 + (unsigned __int64)(unsigned int)v44 > HIDWORD(v44) )
      {
        v33 = (v10 - (__int64)v9) >> 3;
        v34 = v10 - (_QWORD)v9;
        v35 = v10;
        v38 = v9;
        sub_C8D5F0((__int64)&v43, v45, v13 + (unsigned int)v44, 8u, v13, v10);
        v11 = (unsigned int)v44;
        LODWORD(v13) = v33;
        v12 = v34;
        v10 = v35;
        v9 = v38;
      }
      v14 = v43;
      if ( v9 != (const void *)v10 )
      {
        v36 = v13;
        memcpy(&v43[v11], v9, v12);
        LODWORD(v11) = v44;
        v14 = v43;
        LODWORD(v13) = v36;
      }
      LODWORD(v44) = v13 + v11;
      v15 = (unsigned int)(v13 + v11);
    }
    else
    {
      v14 = v43;
    }
    v16 = (_QWORD *)a1[23];
    v42[0] = v14;
    v42[1] = &v14[v15];
    v49 = v49 & 0xF8 | 2;
    v17 = *(__int64 (**)(void))(*v16 + 184LL);
    if ( v17 != sub_31D4830 )
    {
      v27 = v17();
      if ( v27 )
      {
        sub_324BB60(a1, v6, v27);
        goto LABEL_12;
      }
      v16 = (_QWORD *)a1[23];
    }
    v18 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v16[29] + 16LL) + 200LL))(*(_QWORD *)(v16[29] + 16LL));
    sub_3243770((__int64)v46, v18, v42, v40);
LABEL_12:
    sub_3244870(v46, v42);
    if ( v43 != (unsigned __int64 *)v45 )
      _libc_free((unsigned __int64)v43);
  }
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42) <= 1 && *(_DWORD *)(a1[26] + 6224) == 1 )
  {
    v29 = 6;
    if ( v31 )
      v29 = v30;
    LODWORD(v43) = 65547;
    sub_3249A20(a1, (unsigned __int64 **)(a4 + 8), 51, 65547, v29);
  }
  sub_3243D40((__int64)v46);
  sub_3249620(a1, a4, 2, v52);
  if ( v51 )
  {
    LODWORD(v43) = 65547;
    sub_3249A20(a1, (unsigned __int64 **)(a4 + 8), 15875, 65547, v50);
  }
  if ( v47 != &v48 )
    _libc_free((unsigned __int64)v47);
}
