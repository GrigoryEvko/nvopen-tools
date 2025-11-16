// Function: sub_28988E0
// Address: 0x28988e0
//
__int64 __fastcall sub_28988E0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rdi
  _BYTE *v10; // rbx
  __int64 (__fastcall *v11)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v12; // r9
  _BYTE *v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // ebx
  unsigned int v16; // eax
  __int64 v17; // r11
  __int64 v18; // rcx
  __int64 v19; // r9
  unsigned int v20; // ebx
  int v21; // r15d
  unsigned int v22; // r12d
  unsigned int v23; // r13d
  int v24; // ebx
  unsigned int v25; // r15d
  unsigned int v26; // ebx
  __int64 v27; // r13
  int v28; // r12d
  __int64 v29; // rdi
  _DWORD *v30; // r15
  __int64 (__fastcall *v31)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 result; // rax
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // r8
  __int64 v36; // [rsp+8h] [rbp-108h]
  _DWORD *v37; // [rsp+10h] [rbp-100h]
  int v38; // [rsp+18h] [rbp-F8h]
  __int64 v39; // [rsp+18h] [rbp-F8h]
  _BYTE *v40; // [rsp+20h] [rbp-F0h]
  __int64 v41; // [rsp+28h] [rbp-E8h]
  __int64 v42; // [rsp+28h] [rbp-E8h]
  __int64 v43; // [rsp+28h] [rbp-E8h]
  _QWORD *v44; // [rsp+28h] [rbp-E8h]
  __int64 v45; // [rsp+28h] [rbp-E8h]
  __int64 v46; // [rsp+28h] [rbp-E8h]
  _BYTE v47[32]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v48; // [rsp+50h] [rbp-C0h]
  _BYTE v49[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v50; // [rsp+80h] [rbp-90h]
  _DWORD *v51; // [rsp+90h] [rbp-80h] BYREF
  __int64 v52; // [rsp+98h] [rbp-78h]
  _BYTE v53[112]; // [rsp+A0h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v38 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 32LL);
  v48 = 257;
  sub_9B9680((__int64 *)&v51, 0, v38, v7 - v38);
  v37 = v51;
  v36 = (unsigned int)v52;
  v8 = sub_ACADE0(*(__int64 ***)(a3 + 8));
  v9 = *(_QWORD *)(a4 + 80);
  v10 = (_BYTE *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v9 + 112LL);
  if ( v11 != sub_9B6630 )
  {
    v40 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v11)(v9, a3, v10, v37, v36);
LABEL_5:
    if ( v40 )
      goto LABEL_6;
    goto LABEL_35;
  }
  if ( *(_BYTE *)a3 <= 0x15u && *v10 <= 0x15u )
  {
    v40 = (_BYTE *)sub_AD5CE0(a3, (__int64)v10, v37, v36, 0);
    goto LABEL_5;
  }
LABEL_35:
  v50 = 257;
  v33 = sub_BD2C40(112, unk_3F1FE60);
  v40 = v33;
  if ( v33 )
    sub_B4E9E0((__int64)v33, a3, (__int64)v10, v37, v36, (__int64)v49, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
    *(_QWORD *)(a4 + 88),
    v40,
    v47,
    *(_QWORD *)(a4 + 56),
    *(_QWORD *)(a4 + 64));
  sub_94AAF0((unsigned int **)a4, (__int64)v40);
LABEL_6:
  if ( v51 != (_DWORD *)v53 )
    _libc_free((unsigned __int64)v51);
  v51 = v53;
  v52 = 0x1000000000LL;
  if ( a2 )
  {
    v13 = v53;
    v14 = 0;
    v15 = 0;
    while ( 1 )
    {
      *(_DWORD *)&v13[4 * v14] = v15++;
      v14 = (unsigned int)(v52 + 1);
      LODWORD(v52) = v52 + 1;
      if ( a2 == v15 )
        break;
      if ( v14 + 1 > (unsigned __int64)HIDWORD(v52) )
      {
        sub_C8D5F0((__int64)&v51, v53, v14 + 1, 4u, v14 + 1, v12);
        v14 = (unsigned int)v52;
      }
      v13 = v51;
    }
    v16 = v15;
    v17 = (unsigned int)v14;
  }
  else
  {
    v14 = 0;
    v16 = 0;
    v17 = 0;
  }
  v18 = *(_QWORD *)(a1 + 8);
  v19 = *(unsigned int *)(v18 + 32);
  v20 = v38 + a2;
  v21 = v38 + v19;
  if ( v38 + a2 <= v16 )
  {
    v26 = v16;
  }
  else
  {
    v39 = a4;
    v22 = *(_DWORD *)(v18 + 32);
    v23 = v20;
    v24 = v21;
    v25 = v16 + v22 - a2;
    do
    {
      if ( v14 + 1 > (unsigned __int64)HIDWORD(v52) )
      {
        sub_C8D5F0((__int64)&v51, v53, v14 + 1, 4u, v14 + 1, v19);
        v14 = (unsigned int)v52;
      }
      v51[v14] = v25++;
      v14 = (unsigned int)(v52 + 1);
      LODWORD(v52) = v52 + 1;
    }
    while ( v24 != v25 );
    v19 = v22;
    a4 = v39;
    v26 = v23;
    v17 = (unsigned int)v14;
  }
  if ( v26 < (unsigned int)v19 )
  {
    v27 = a4;
    v28 = v19;
    do
    {
      if ( v14 + 1 > (unsigned __int64)HIDWORD(v52) )
      {
        sub_C8D5F0((__int64)&v51, v53, v14 + 1, 4u, v14 + 1, v19);
        v14 = (unsigned int)v52;
      }
      v51[v14] = v26++;
      v14 = (unsigned int)(v52 + 1);
      LODWORD(v52) = v52 + 1;
    }
    while ( v28 != v26 );
    a4 = v27;
    v17 = (unsigned int)v14;
  }
  v29 = *(_QWORD *)(a4 + 80);
  v30 = v51;
  v48 = 257;
  v31 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v29 + 112LL);
  if ( v31 != sub_9B6630 )
  {
    v46 = v17;
    result = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v31)(v29, a1, v40, v51, v17);
    v17 = v46;
LABEL_31:
    if ( result )
      goto LABEL_32;
    goto LABEL_38;
  }
  if ( *(_BYTE *)a1 <= 0x15u && *v40 <= 0x15u )
  {
    v41 = v17;
    result = sub_AD5CE0(a1, (__int64)v40, v51, v17, 0);
    v17 = v41;
    goto LABEL_31;
  }
LABEL_38:
  v43 = v17;
  v50 = 257;
  v34 = sub_BD2C40(112, unk_3F1FE60);
  if ( v34 )
  {
    v35 = v43;
    v44 = v34;
    sub_B4E9E0((__int64)v34, a1, (__int64)v40, v30, v35, (__int64)v49, 0, 0);
    v34 = v44;
  }
  v45 = (__int64)v34;
  (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
    *(_QWORD *)(a4 + 88),
    v34,
    v47,
    *(_QWORD *)(a4 + 56),
    *(_QWORD *)(a4 + 64));
  sub_94AAF0((unsigned int **)a4, v45);
  result = v45;
LABEL_32:
  if ( v51 != (_DWORD *)v53 )
  {
    v42 = result;
    _libc_free((unsigned __int64)v51);
    return v42;
  }
  return result;
}
