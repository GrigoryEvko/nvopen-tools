// Function: sub_2233080
// Address: 0x2233080
//
__int64 __fastcall sub_2233080(__int64 a1, __int64 a2, char a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v10; // rax
  int v11; // r11d
  unsigned __int64 v12; // rsi
  __int64 v13; // r13
  int v14; // eax
  unsigned int v15; // r11d
  char *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r12
  char v20; // dl
  int v21; // eax
  void *v22; // rsp
  char v23; // dl
  char v24; // cl
  __int64 v25; // rdx
  void *v26; // rsp
  char *v27; // rax
  char v28; // dl
  char v29; // dl
  _BYTE v30[2]; // [rsp-Eh] [rbp-A0h] BYREF
  _BYTE v31[13]; // [rsp-Ch] [rbp-9Eh] BYREF
  _BYTE v32[20]; // [rsp+1Ah] [rbp-78h] BYREF
  int v33; // [rsp+2Eh] [rbp-64h]
  __int64 v34; // [rsp+32h] [rbp-60h]
  _BYTE *v35; // [rsp+3Ah] [rbp-58h]
  int v36; // [rsp+42h] [rbp-50h]
  char v37; // [rsp+49h] [rbp-49h]
  char *v38; // [rsp+4Ah] [rbp-48h]
  char v39; // [rsp+5Dh] [rbp-35h] BYREF
  int v40[13]; // [rsp+5Eh] [rbp-34h] BYREF

  v34 = a1;
  v36 = a5;
  v10 = sub_2232A70((__int64)&v39, (__int64 *)(a4 + 208));
  v11 = *(_DWORD *)(a4 + 24);
  v12 = a6;
  v13 = v10;
  v33 = v11 & 0x4A;
  if ( a6 <= 0 )
  {
    v12 = -a6;
    if ( v33 == 8 || v33 == 64 )
      v12 = a6;
  }
  v35 = v30;
  v37 = v33 != 8 && v33 != 64;
  LODWORD(v38) = v11;
  v14 = sub_22329C0(v32, v12, v10 + 74, v11, v37);
  v15 = (unsigned int)v38;
  v40[0] = v14;
  v16 = &v35[40 - v14];
  if ( !*(_BYTE *)(v13 + 32) )
  {
    if ( v37 )
      goto LABEL_6;
LABEL_16:
    if ( (v15 & 0x200) != 0 && a6 )
    {
      if ( v33 == 64 )
      {
        v29 = *(_BYTE *)(v13 + 78);
        ++v14;
        --v16;
        v40[0] = v14;
      }
      else
      {
        v14 += 2;
        v16 -= 2;
        v28 = *(_BYTE *)(v13 + ((v15 >> 14) & 1) + 76);
        v40[0] = v14;
        v16[1] = v28;
        v29 = *(_BYTE *)(v13 + 78);
      }
      *v16 = v29;
    }
    goto LABEL_8;
  }
  v24 = *(_BYTE *)(v13 + 73);
  v25 = *(_QWORD *)(v13 + 24);
  LODWORD(v35) = (_DWORD)v38;
  v26 = alloca(2LL * (v14 + 1) + 8);
  v27 = *(char **)(v13 + 16);
  v38 = v31;
  sub_22316B0(v34, v27, v25, v24, a4, v31, (__int64)v16, v40);
  v14 = v40[0];
  v15 = (unsigned int)v35;
  v16 = v38;
  if ( !v37 )
    goto LABEL_16;
LABEL_6:
  if ( a6 < 0 )
  {
    v23 = *(_BYTE *)(v13 + 74);
    ++v14;
    --v16;
    v40[0] = v14;
    *v16 = v23;
  }
  else if ( (v15 & 0x800) != 0 )
  {
    v20 = *(_BYTE *)(v13 + 75);
    v21 = v14 + 1;
    --v16;
    v18 = v21;
    v40[0] = v21;
    *v16 = v20;
    v17 = *(_QWORD *)(a4 + 16);
    if ( v21 >= v17 )
      goto LABEL_9;
    goto LABEL_13;
  }
LABEL_8:
  v17 = *(_QWORD *)(a4 + 16);
  v18 = v14;
  if ( v14 >= v17 )
    goto LABEL_9;
LABEL_13:
  v22 = alloca(v17 + 8);
  sub_22328D0(v34, v36, v17, a4, v30, v16, v40);
  v18 = v40[0];
  v16 = v30;
LABEL_9:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)a2 + 96LL))(a2, v16, v18);
  return a2;
}
