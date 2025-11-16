// Function: sub_2232E00
// Address: 0x2232e00
//
__int64 __fastcall sub_2232E00(__int64 a1, __int64 a2, char a3, __int64 a4, int a5, unsigned __int64 a6)
{
  __int64 v10; // r12
  int v11; // eax
  unsigned int v12; // r11d
  char *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r12
  char v17; // cl
  __int64 v18; // rdx
  void *v19; // rsp
  char *v20; // rax
  char v21; // dl
  char v22; // dl
  void *v23; // rsp
  _BYTE v24[2]; // [rsp-Eh] [rbp-A0h] BYREF
  _BYTE v25[13]; // [rsp-Ch] [rbp-9Eh] BYREF
  _BYTE v26[20]; // [rsp+1Ah] [rbp-78h] BYREF
  int v27; // [rsp+2Eh] [rbp-64h]
  __int64 v28; // [rsp+32h] [rbp-60h]
  _BYTE *v29; // [rsp+3Ah] [rbp-58h]
  int v30; // [rsp+42h] [rbp-50h]
  char v31; // [rsp+49h] [rbp-49h]
  char *v32; // [rsp+4Ah] [rbp-48h]
  char v33; // [rsp+5Dh] [rbp-35h] BYREF
  int v34[13]; // [rsp+5Eh] [rbp-34h] BYREF

  v28 = a1;
  v30 = a5;
  v10 = sub_2232A70((__int64)&v33, (__int64 *)(a4 + 208));
  LODWORD(v32) = *(_DWORD *)(a4 + 24);
  v29 = v24;
  v27 = (unsigned __int8)v32 & 0x4A;
  v31 = v27 != 8 && v27 != 64;
  v11 = sub_2232910(v26, a6, v10 + 74, (__int16)v32, v31);
  v12 = (unsigned int)v32;
  v34[0] = v11;
  v13 = &v29[40 - v11];
  if ( *(_BYTE *)(v10 + 32) )
  {
    v17 = *(_BYTE *)(v10 + 73);
    v18 = *(_QWORD *)(v10 + 24);
    LODWORD(v29) = (_DWORD)v32;
    v19 = alloca(2LL * (v11 + 1) + 8);
    v20 = *(char **)(v10 + 16);
    v32 = v25;
    sub_22316B0(v28, v20, v18, v17, a4, v25, (__int64)v13, v34);
    v11 = v34[0];
    v12 = (unsigned int)v29;
    v13 = v32;
    if ( v31 )
      goto LABEL_3;
  }
  else if ( v31 )
  {
    goto LABEL_3;
  }
  if ( (v12 & 0x200) != 0 && a6 )
  {
    if ( v27 == 64 )
    {
      v22 = *(_BYTE *)(v10 + 78);
      ++v11;
      --v13;
      v34[0] = v11;
    }
    else
    {
      v11 += 2;
      v13 -= 2;
      v21 = *(_BYTE *)(v10 + ((v12 >> 14) & 1) + 76);
      v34[0] = v11;
      v13[1] = v21;
      v22 = *(_BYTE *)(v10 + 78);
    }
    *v13 = v22;
  }
LABEL_3:
  v14 = *(_QWORD *)(a4 + 16);
  v15 = v11;
  if ( v11 < v14 )
  {
    v23 = alloca(v14 + 8);
    sub_22328D0(v28, v30, v14, a4, v24, v13, v34);
    v15 = v34[0];
    v13 = v24;
  }
  *(_QWORD *)(a4 + 16) = 0;
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)a2 + 96LL))(a2, v13, v15);
  return a2;
}
