// Function: sub_2246400
// Address: 0x2246400
//
__int64 __fastcall sub_2246400(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5, __int64 a6)
{
  __int64 v10; // rax
  int v11; // r11d
  unsigned __int64 v12; // rsi
  __int64 v13; // r13
  int v14; // eax
  unsigned int v15; // r11d
  bool v16; // zf
  wchar_t *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r12
  wchar_t v21; // edx
  int v22; // eax
  void *v23; // rsp
  wchar_t v24; // edx
  int v25; // ecx
  __int64 v26; // rdx
  void *v27; // rsp
  char *v28; // rax
  wchar_t v29; // edx
  wchar_t v30; // edx
  wchar_t v31; // [rsp-Eh] [rbp-120h] BYREF
  int v32; // [rsp-6h] [rbp-118h] BYREF
  _DWORD v33[5]; // [rsp+92h] [rbp-80h] BYREF
  int v34; // [rsp+A6h] [rbp-6Ch]
  __int64 v35; // [rsp+AAh] [rbp-68h]
  wchar_t *v36; // [rsp+B2h] [rbp-60h]
  unsigned int v37; // [rsp+BAh] [rbp-58h]
  char v38; // [rsp+C1h] [rbp-51h]
  wchar_t *v39; // [rsp+C2h] [rbp-50h]
  __int64 v40; // [rsp+CAh] [rbp-48h]
  char v41; // [rsp+DDh] [rbp-35h] BYREF
  int v42[13]; // [rsp+DEh] [rbp-34h] BYREF

  v35 = a1;
  v37 = a5;
  v10 = sub_22462F0((__int64)&v41, (__int64 *)(a4 + 208));
  v11 = *(_DWORD *)(a4 + 24);
  v12 = a6;
  v13 = v10;
  v40 = v10 + 80;
  v34 = v11 & 0x4A;
  if ( a6 <= 0 )
  {
    v12 = -a6;
    if ( (v11 & 0x4A) == 8 || (v11 & 0x4A) == 64 )
      v12 = a6;
  }
  v36 = &v31;
  v38 = (v11 & 0x4A) != 8 && (v11 & 0x4A) != 64;
  LODWORD(v39) = v11;
  v14 = sub_2246150(v33, v12, v40, v11, v38);
  v15 = (unsigned int)v39;
  v16 = *(_BYTE *)(v13 + 32) == 0;
  v42[0] = v14;
  v17 = &v36[40 - v14];
  if ( v16 )
  {
    if ( v38 )
      goto LABEL_6;
LABEL_16:
    if ( (v15 & 0x200) != 0 && a6 )
    {
      if ( v34 == 64 )
      {
        v30 = *(_DWORD *)(v13 + 96);
        ++v14;
        --v17;
        v42[0] = v14;
      }
      else
      {
        v14 += 2;
        v17 -= 2;
        v29 = *(_DWORD *)(v40 + 4LL * ((v15 >> 14) & 1) + 8);
        v42[0] = v14;
        v17[1] = v29;
        v30 = *(_DWORD *)(v13 + 96);
      }
      *v17 = v30;
    }
    goto LABEL_8;
  }
  v25 = *(_DWORD *)(v13 + 76);
  v26 = *(_QWORD *)(v13 + 24);
  LODWORD(v36) = (_DWORD)v39;
  v27 = alloca(8LL * (v14 + 1) + 8);
  v28 = *(char **)(v13 + 16);
  v39 = &v32;
  sub_2244F80(v35, v28, v26, v25, a4, &v32, (__int64)v17, v42);
  v14 = v42[0];
  v15 = (unsigned int)v36;
  v17 = v39;
  if ( !v38 )
    goto LABEL_16;
LABEL_6:
  if ( a6 < 0 )
  {
    v24 = *(_DWORD *)(v13 + 80);
    ++v14;
    --v17;
    v42[0] = v14;
    *v17 = v24;
  }
  else if ( (v15 & 0x800) != 0 )
  {
    v21 = *(_DWORD *)(v13 + 84);
    v22 = v14 + 1;
    --v17;
    v19 = v22;
    v42[0] = v22;
    *v17 = v21;
    v18 = *(_QWORD *)(a4 + 16);
    if ( v22 >= v18 )
      goto LABEL_9;
    goto LABEL_13;
  }
LABEL_8:
  v18 = *(_QWORD *)(a4 + 16);
  v19 = v14;
  if ( v14 >= v18 )
    goto LABEL_9;
LABEL_13:
  v23 = alloca(4 * v18 + 8);
  sub_2246120(v35, v37, v18, a4, &v31, v17, v42);
  v19 = v42[0];
  v17 = &v31;
LABEL_9:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, wchar_t *, __int64))(*(_QWORD *)a2 + 96LL))(a2, v17, v19);
  return a2;
}
