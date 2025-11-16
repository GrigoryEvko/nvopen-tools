// Function: sub_2247460
// Address: 0x2247460
//
__int64 __fastcall sub_2247460(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        char a6,
        __int64 a7,
        wchar_t *a8)
{
  __int64 v11; // rax
  __int64 v12; // r13
  wchar_t *v13; // rsi
  __int64 v14; // r9
  int v15; // eax
  __int64 *v16; // rdi
  __int64 *v17; // rdi
  void *v18; // rsp
  __int64 v19; // rax
  wchar_t *v20; // r15
  __int64 v21; // r13
  __int64 v22; // rcx
  wchar_t *v23; // rax
  wchar_t *v24; // r8
  char v25; // si
  __int64 v26; // r11
  void *v28; // rsp
  void *v29; // rsp
  wchar_t *v30; // r9
  int v31; // ebx
  void *v32; // rsp
  int v33; // eax
  __int64 *v34; // rdi
  void *v35; // rsp
  wchar_t v36; // [rsp-Eh] [rbp-D0h] BYREF
  int v37; // [rsp-Ah] [rbp-CCh] BYREF
  __int64 *v38; // [rsp+3Ah] [rbp-88h]
  unsigned int v39; // [rsp+46h] [rbp-7Ch]
  __int64 *v40; // [rsp+4Ah] [rbp-78h]
  __int64 v41; // [rsp+52h] [rbp-70h]
  __int64 v42; // [rsp+5Ah] [rbp-68h]
  wchar_t *v43; // [rsp+62h] [rbp-60h]
  __int64 v44; // [rsp+6Ah] [rbp-58h]
  char v45; // [rsp+75h] [rbp-4Dh] BYREF
  int v46; // [rsp+76h] [rbp-4Ch] BYREF
  __int64 v47; // [rsp+7Ah] [rbp-48h] BYREF
  char v48[64]; // [rsp+82h] [rbp-40h] BYREF

  v41 = a1;
  v42 = a3;
  v39 = a5;
  v43 = (wchar_t *)(a4 + 208);
  v11 = sub_22462F0((__int64)&v45, (__int64 *)(a4 + 208));
  v12 = *(_QWORD *)(a4 + 8);
  v44 = v11;
  if ( v12 < 0 )
    v12 = 6;
  sub_2255110(a4, v48, (unsigned int)a6);
  if ( (*(_DWORD *)(a4 + 24) & 0x104) == 0x104 )
  {
    v13 = &v36;
    v47 = sub_2208E60(a4, v48);
    v40 = &v47;
    v33 = sub_2218500((__int64)&v47, (char *)&v36, 54, v48);
    v34 = v40;
    v46 = v33;
    if ( v33 > 53 )
    {
      v38 = v40;
      LODWORD(v40) = v33 + 1;
      v35 = alloca(v33 + 1 + 8LL);
      v47 = sub_2208E60(v34, &v36);
      v13 = &v36;
      v46 = sub_2218500((__int64)v38, (char *)&v36, (int)v40, v48);
    }
  }
  else
  {
    v13 = &v36;
    v47 = sub_2208E60(a4, v48);
    v40 = &v47;
    v15 = sub_2218500((__int64)&v47, (char *)&v36, 54, v48, v12, v14, a7, a8);
    v16 = v40;
    v46 = v15;
    if ( v15 > 53 )
    {
      v38 = v40;
      LODWORD(v40) = v15 + 1;
      v28 = alloca(v15 + 1 + 8LL);
      v47 = sub_2208E60(v16, &v36);
      v13 = a8;
      v46 = sub_2218500((__int64)v38, (char *)&v36, (int)v40, v48, v12);
    }
  }
  v17 = (__int64 *)sub_2243120(v43, (__int64)v13);
  v18 = alloca(4LL * v46 + 8);
  v19 = *v17;
  v20 = &v36;
  v43 = &v36;
  (*(void (__fastcall **)(__int64 *, wchar_t *, char *, wchar_t *))(v19 + 88))(v17, &v36, (char *)&v36 + v46, &v36);
  v21 = v46;
  v22 = v46;
  if ( !v46 )
  {
    if ( *(_BYTE *)(v44 + 32) )
    {
      v24 = 0;
      goto LABEL_15;
    }
LABEL_9:
    v26 = *(_QWORD *)(a4 + 16);
    if ( v21 >= v26 )
      goto LABEL_10;
    goto LABEL_18;
  }
  LODWORD(v40) = v46;
  v23 = (wchar_t *)memchr(&v36, 46, v46);
  v22 = (unsigned int)v40;
  v24 = v23;
  v25 = *(_BYTE *)(v44 + 32);
  if ( !v23 )
  {
    if ( v25 && ((int)v40 <= 2 || SBYTE1(v36) <= 57 && (unsigned __int8)(BYTE2(v36) - 48) <= 9u && SBYTE1(v36) > 47) )
      goto LABEL_15;
    goto LABEL_9;
  }
  v24 = &v43[(char *)v23 - (char *)&v36];
  *v24 = *(_DWORD *)(v44 + 72);
  if ( !v25 )
    goto LABEL_9;
LABEL_15:
  v29 = alloca(8 * v21 + 8);
  if ( (((_BYTE)v36 - 43) & 0xFD) != 0 )
  {
    v30 = &v36;
    v31 = 0;
  }
  else
  {
    v30 = &v37;
    v46 = v22 - 1;
    v31 = 1;
    v20 = v43 + 1;
    v36 = *v43;
  }
  sub_2244ED0(v41, *(char **)(v44 + 16), *(_QWORD *)(v44 + 24), *(_DWORD *)(v44 + 76), v24, v30, (__int64)v20, &v46);
  v26 = *(_QWORD *)(a4 + 16);
  v22 = (unsigned int)(v31 + v46);
  v21 = (int)v22;
  v46 += v31;
  if ( (int)v22 >= v26 )
    goto LABEL_10;
LABEL_18:
  v32 = alloca(4 * v26 + 8);
  sub_2246120(v41, v39, v26, a4, &v36, &v36, &v46);
  v21 = v46;
LABEL_10:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !(_BYTE)v42 )
    (*(__int64 (__fastcall **)(__int64, wchar_t *, __int64, __int64))(*(_QWORD *)a2 + 96LL))(a2, &v36, v21, v22);
  return a2;
}
