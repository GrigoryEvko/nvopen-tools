// Function: sub_2246E30
// Address: 0x2246e30
//
__int64 __fastcall sub_2246E30(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5, unsigned __int64 a6)
{
  __int64 v10; // rax
  int v11; // r10d
  __int64 v12; // r12
  int v13; // eax
  unsigned int v14; // r10d
  bool v15; // zf
  wchar_t *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r12
  int v20; // ecx
  char *v21; // r11
  __int64 v22; // rdx
  void *v23; // rsp
  wchar_t v24; // edx
  wchar_t v25; // edx
  void *v26; // rsp
  wchar_t v27; // [rsp-Eh] [rbp-120h] BYREF
  int v28; // [rsp-6h] [rbp-118h] BYREF
  _DWORD v29[4]; // [rsp+92h] [rbp-80h] BYREF
  __int64 v30; // [rsp+A2h] [rbp-70h]
  int v31; // [rsp+AEh] [rbp-64h]
  __int64 v32; // [rsp+B2h] [rbp-60h]
  wchar_t *v33; // [rsp+BAh] [rbp-58h]
  wchar_t *v34; // [rsp+C2h] [rbp-50h]
  unsigned int v35; // [rsp+CAh] [rbp-48h]
  char v36; // [rsp+D1h] [rbp-41h]
  char v37; // [rsp+DDh] [rbp-35h] BYREF
  int v38[13]; // [rsp+DEh] [rbp-34h] BYREF

  v32 = a1;
  v35 = a5;
  v10 = sub_22462F0((__int64)&v37, (__int64 *)(a4 + 208));
  v11 = *(_DWORD *)(a4 + 24);
  v12 = v10;
  v30 = v10 + 80;
  v33 = &v27;
  v31 = v11 & 0x4A;
  LODWORD(v34) = v11;
  v36 = v31 != 8 && v31 != 64;
  v13 = sub_2246220(v29, a6, v10 + 80, v11, v36);
  v14 = (unsigned int)v34;
  v15 = *(_BYTE *)(v12 + 32) == 0;
  v38[0] = v13;
  v16 = &v33[40 - v13];
  if ( v15 )
  {
    if ( v36 )
      goto LABEL_3;
  }
  else
  {
    v20 = *(_DWORD *)(v12 + 76);
    v21 = *(char **)(v12 + 16);
    v22 = *(_QWORD *)(v12 + 24);
    LODWORD(v33) = (_DWORD)v34;
    v23 = alloca(8LL * (v13 + 1) + 8);
    v34 = &v28;
    sub_2244F80(v32, v21, v22, v20, a4, &v28, (__int64)v16, v38);
    v13 = v38[0];
    v14 = (unsigned int)v33;
    v16 = v34;
    if ( v36 )
      goto LABEL_3;
  }
  if ( (v14 & 0x200) != 0 && a6 )
  {
    if ( v31 == 64 )
    {
      v25 = *(_DWORD *)(v12 + 96);
      ++v13;
      --v16;
      v38[0] = v13;
    }
    else
    {
      v13 += 2;
      v16 -= 2;
      v24 = *(_DWORD *)(v30 + 4LL * ((v14 >> 14) & 1) + 8);
      v38[0] = v13;
      v16[1] = v24;
      v25 = *(_DWORD *)(v12 + 96);
    }
    *v16 = v25;
  }
LABEL_3:
  v17 = *(_QWORD *)(a4 + 16);
  v18 = v13;
  if ( v13 < v17 )
  {
    v26 = alloca(4 * v17 + 8);
    sub_2246120(v32, v35, v17, a4, &v27, v16, v38);
    v18 = v38[0];
    v16 = &v27;
  }
  *(_QWORD *)(a4 + 16) = 0;
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, wchar_t *, __int64))(*(_QWORD *)a2 + 96LL))(a2, v16, v18);
  return a2;
}
