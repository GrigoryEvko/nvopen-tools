// Function: sub_B0B320
// Address: 0xb0b320
//
__int64 __fastcall sub_B0B320(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int8 a5,
        __int64 a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // r10
  __int64 *v10; // r12
  __int16 v11; // bx
  __int64 v12; // r11
  int v13; // r15d
  int v14; // eax
  int v15; // r8d
  __int64 v16; // r15
  __int64 *v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 *v22; // r15
  __int64 result; // rax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r15
  char v28; // r8
  int v29; // [rsp+0h] [rbp-A0h]
  int v30; // [rsp+4h] [rbp-9Ch]
  __int64 v31; // [rsp+8h] [rbp-98h]
  int v32; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  __int64 v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h]
  __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  __int64 v40; // [rsp+48h] [rbp-58h] BYREF
  __int64 v41; // [rsp+50h] [rbp-50h] BYREF
  __int8 v42[8]; // [rsp+58h] [rbp-48h] BYREF
  __int64 v43[8]; // [rsp+60h] [rbp-40h] BYREF

  v8 = a4;
  v10 = a1;
  v11 = a2;
  if ( a7 )
    goto LABEL_17;
  v12 = *a1;
  LODWORD(v39) = a2;
  v40 = a3;
  v41 = a4;
  v42[0] = a5;
  v43[0] = a6;
  v13 = *(_DWORD *)(v12 + 1264);
  v36 = v12;
  v37 = *(_QWORD *)(v12 + 1248);
  if ( !v13 )
    goto LABEL_16;
  v31 = a6;
  v14 = sub_AF9230((int *)&v39, &v40, &v41, v42, v43);
  v15 = v13 - 1;
  v30 = 1;
  v35 = (v13 - 1) & v14;
  v16 = a4;
  v29 = v15;
  v34 = v31;
  while ( 1 )
  {
    v17 = (__int64 *)(v37 + 8LL * v35);
    v18 = *v17;
    if ( *v17 == -4096 )
    {
LABEL_21:
      v10 = a1;
      v11 = a2;
      v8 = v16;
      a6 = v34;
      goto LABEL_16;
    }
    if ( v18 != -8192 )
      break;
LABEL_8:
    if ( v18 == -4096 )
      goto LABEL_21;
    v35 = v29 & (v30 + v35);
    ++v30;
  }
  v32 = v39;
  if ( v32 != (unsigned __int16)sub_AF18C0(*v17)
    || (v19 = sub_AF5140(v18, 0), v40 != v19)
    || (v20 = sub_A17150((_BYTE *)(v18 - 16)), v41 != *((_QWORD *)v20 + 1))
    || v42[0] != *(_BYTE *)(v18 + 1) >> 7
    || (v21 = sub_A17150((_BYTE *)(v18 - 16)), v43[0] != *((_QWORD *)v21 + 2)) )
  {
    v18 = *v17;
    goto LABEL_8;
  }
  v8 = v16;
  v22 = (__int64 *)(v37 + 8LL * v35);
  v10 = a1;
  v11 = a2;
  a6 = v34;
  if ( v22 == (__int64 *)(*(_QWORD *)(v36 + 1248) + 8LL * *(unsigned int *)(v36 + 1264)) || (result = *v22) == 0 )
  {
LABEL_16:
    result = 0;
    if ( a8 )
    {
LABEL_17:
      v24 = *v10;
      v39 = a3;
      v40 = v8;
      v41 = a6;
      v25 = v24 + 1240;
      v26 = sub_B97910(16, 3, a7);
      v27 = v26;
      if ( v26 )
      {
        sub_B971C0(v26, (_DWORD)v10, 24, a7, (unsigned int)&v39, 3, 0, 0);
        v28 = *(_BYTE *)(v27 + 1);
        *(_WORD *)(v27 + 2) = v11;
        *(_BYTE *)(v27 + 1) = (a5 << 7) | v28 & 0x7F;
      }
      return sub_B0B0F0(v27, a7, v25);
    }
  }
  return result;
}
