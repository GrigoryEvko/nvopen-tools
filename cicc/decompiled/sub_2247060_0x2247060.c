// Function: sub_2247060
// Address: 0x2247060
//
__int64 __fastcall sub_2247060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, char a6, double a7)
{
  __int64 v10; // rax
  __int64 v11; // r13
  int v12; // eax
  __int64 *v13; // rdi
  __int64 *v14; // rdi
  void *v15; // rsp
  __int64 v16; // rax
  wchar_t *v17; // r15
  __int64 v18; // r13
  __int64 v19; // rcx
  wchar_t *v20; // rax
  wchar_t *v21; // r8
  char v22; // si
  __int64 v23; // r11
  void *v25; // rsp
  void *v26; // rsp
  wchar_t *v27; // r9
  int v28; // ebx
  void *v29; // rsp
  int v30; // eax
  __int64 *v31; // rdi
  void *v32; // rsp
  wchar_t v33; // [rsp-Eh] [rbp-D0h] BYREF
  int v34; // [rsp-Ah] [rbp-CCh] BYREF
  __int64 *v35; // [rsp+32h] [rbp-90h]
  __int64 *v36; // [rsp+3Ah] [rbp-88h]
  unsigned int v37; // [rsp+46h] [rbp-7Ch]
  __int64 v38; // [rsp+4Ah] [rbp-78h]
  __int64 v39; // [rsp+52h] [rbp-70h]
  _QWORD *v40; // [rsp+5Ah] [rbp-68h]
  double v41; // [rsp+62h] [rbp-60h]
  __int64 v42; // [rsp+6Ah] [rbp-58h]
  char v43; // [rsp+75h] [rbp-4Dh] BYREF
  int v44; // [rsp+76h] [rbp-4Ch] BYREF
  __int64 v45; // [rsp+7Ah] [rbp-48h] BYREF
  char v46[64]; // [rsp+82h] [rbp-40h] BYREF

  v38 = a1;
  v39 = a3;
  v37 = a5;
  v41 = a7;
  v40 = (_QWORD *)(a4 + 208);
  v10 = sub_22462F0((__int64)&v43, (__int64 *)(a4 + 208));
  v11 = *(_QWORD *)(a4 + 8);
  v42 = v10;
  if ( v11 < 0 )
    v11 = 6;
  sub_2255110(a4, v46, (unsigned int)a6);
  if ( (*(_DWORD *)(a4 + 24) & 0x104) == 0x104 )
  {
    v45 = sub_2208E60(a4, v46);
    v36 = &v45;
    v30 = sub_2218500((__int64)&v45, (char *)&v33, 45, v46, v41);
    v31 = v36;
    v44 = v30;
    if ( v30 > 44 )
    {
      v35 = v36;
      LODWORD(v36) = v30 + 1;
      v32 = alloca(v30 + 1 + 8LL);
      v45 = sub_2208E60(v31, &v33);
      v44 = sub_2218500((__int64)v35, (char *)&v33, (int)v36, v46, v41);
    }
  }
  else
  {
    v45 = sub_2208E60(a4, v46);
    v36 = &v45;
    v12 = sub_2218500((__int64)&v45, (char *)&v33, 45, v46, v11, v41);
    v13 = v36;
    v44 = v12;
    if ( v12 > 44 )
    {
      v35 = v36;
      LODWORD(v36) = v12 + 1;
      v25 = alloca(v12 + 1 + 8LL);
      v45 = sub_2208E60(v13, &v33);
      v44 = sub_2218500((__int64)v35, (char *)&v33, (int)v36, v46, v11, v41);
    }
  }
  v14 = (__int64 *)sub_2243120(v40, (__int64)&v33);
  v15 = alloca(4LL * v44 + 8);
  v16 = *v14;
  v17 = &v33;
  v41 = COERCE_DOUBLE(&v33);
  (*(void (__fastcall **)(__int64 *, wchar_t *, char *, wchar_t *))(v16 + 88))(v14, &v33, (char *)&v33 + v44, &v33);
  v18 = v44;
  v19 = v44;
  if ( !v44 )
  {
    if ( *(_BYTE *)(v42 + 32) )
    {
      v21 = 0;
      goto LABEL_15;
    }
LABEL_9:
    v23 = *(_QWORD *)(a4 + 16);
    if ( v18 >= v23 )
      goto LABEL_10;
    goto LABEL_18;
  }
  LODWORD(v40) = v44;
  v20 = (wchar_t *)memchr(&v33, 46, v44);
  v19 = (unsigned int)v40;
  v21 = v20;
  v22 = *(_BYTE *)(v42 + 32);
  if ( !v20 )
  {
    if ( v22 && ((int)v40 <= 2 || SBYTE1(v33) <= 57 && (unsigned __int8)(BYTE2(v33) - 48) <= 9u && SBYTE1(v33) > 47) )
      goto LABEL_15;
    goto LABEL_9;
  }
  v21 = (wchar_t *)(*(_QWORD *)&v41 + 4 * ((char *)v20 - (char *)&v33));
  *v21 = *(_DWORD *)(v42 + 72);
  if ( !v22 )
    goto LABEL_9;
LABEL_15:
  v26 = alloca(8 * v18 + 8);
  if ( (((_BYTE)v33 - 43) & 0xFD) != 0 )
  {
    v27 = &v33;
    v28 = 0;
  }
  else
  {
    v27 = &v34;
    v44 = v19 - 1;
    v28 = 1;
    v17 = (wchar_t *)(*(_QWORD *)&v41 + 4LL);
    v33 = **(_DWORD **)&v41;
  }
  sub_2244ED0(v38, *(char **)(v42 + 16), *(_QWORD *)(v42 + 24), *(_DWORD *)(v42 + 76), v21, v27, (__int64)v17, &v44);
  v23 = *(_QWORD *)(a4 + 16);
  v19 = (unsigned int)(v28 + v44);
  v18 = (int)v19;
  v44 += v28;
  if ( (int)v19 >= v23 )
    goto LABEL_10;
LABEL_18:
  v29 = alloca(4 * v23 + 8);
  sub_2246120(v38, v37, v23, a4, &v33, &v33, &v44);
  v18 = v44;
LABEL_10:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !(_BYTE)v39 )
    (*(__int64 (__fastcall **)(__int64, wchar_t *, __int64, __int64))(*(_QWORD *)a2 + 96LL))(a2, &v33, v18, v19);
  return a2;
}
