// Function: sub_B0FC60
// Address: 0xb0fc60
//
__int64 __fastcall sub_B0FC60(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        char a10)
{
  __int64 v10; // r10
  __int64 v11; // r13
  _QWORD *v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r11
  int v15; // r15d
  unsigned int i; // ebx
  __int64 *v17; // r12
  __int64 v18; // r13
  int v19; // r15d
  unsigned int v20; // r9d
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // rax
  _BYTE *v25; // rax
  __int64 *v26; // rcx
  __int64 result; // rax
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // r15
  int v31; // [rsp+Ch] [rbp-B4h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+38h] [rbp-88h]
  int v37; // [rsp+40h] [rbp-80h]
  __int64 v39; // [rsp+50h] [rbp-70h] BYREF
  __int64 v40; // [rsp+58h] [rbp-68h] BYREF
  __int64 v41; // [rsp+60h] [rbp-60h] BYREF
  __int64 v42; // [rsp+68h] [rbp-58h] BYREF
  __int64 v43; // [rsp+70h] [rbp-50h] BYREF
  __int64 v44; // [rsp+78h] [rbp-48h] BYREF
  __int64 v45[8]; // [rsp+80h] [rbp-40h] BYREF

  v10 = a5;
  v11 = a4;
  v12 = a1;
  v13 = a3;
  if ( a9 )
    goto LABEL_19;
  v14 = *a1;
  v40 = a3;
  v41 = a4;
  LODWORD(v39) = a2;
  v42 = a5;
  LODWORD(v43) = a6;
  v44 = a7;
  v45[0] = a8;
  v15 = *(_DWORD *)(v14 + 1424);
  v36 = *(_QWORD *)(v14 + 1408);
  if ( !v15 )
    goto LABEL_18;
  v37 = 1;
  v31 = v15 - 1;
  v32 = v14;
  for ( i = (v15 - 1) & sub_AFB320((int *)&v39, &v40, &v41, &v42, (int *)&v43, &v44, v45); ; i = v20 & v31 )
  {
    v17 = (__int64 *)(v36 + 8LL * i);
    v18 = *v17;
    if ( *v17 == -4096 )
    {
LABEL_23:
      v12 = a1;
      v13 = a3;
      v11 = a4;
      v10 = a5;
      goto LABEL_18;
    }
    if ( v18 != -8192 )
      break;
LABEL_8:
    if ( v18 == -4096 )
      goto LABEL_23;
    v20 = i + v37++;
  }
  v19 = v39;
  if ( v19 != (unsigned __int16)sub_AF18C0(*v17)
    || (v21 = sub_A17150((_BYTE *)(v18 - 16)), v40 != *(_QWORD *)v21)
    || (v22 = sub_A17150((_BYTE *)(v18 - 16)), v41 != *((_QWORD *)v22 + 1))
    || (v23 = sub_A17150((_BYTE *)(v18 - 16)), v42 != *((_QWORD *)v23 + 3))
    || (_DWORD)v43 != *(_DWORD *)(v18 + 4)
    || (v24 = sub_AF5140(v18, 2u), v44 != v24)
    || (v25 = sub_A17150((_BYTE *)(v18 - 16)), v45[0] != *((_QWORD *)v25 + 4)) )
  {
    v18 = *v17;
    goto LABEL_8;
  }
  v26 = (__int64 *)(v36 + 8LL * i);
  v13 = a3;
  v12 = a1;
  v11 = a4;
  v10 = a5;
  if ( v26 == (__int64 *)(*(_QWORD *)(v32 + 1408) + 8LL * *(unsigned int *)(v32 + 1424)) || (result = *v26) == 0 )
  {
LABEL_18:
    result = 0;
    if ( a10 )
    {
LABEL_19:
      v40 = v11;
      v39 = v13;
      v41 = a7;
      v42 = v10;
      v43 = a8;
      v28 = *v12 + 1400LL;
      v29 = sub_B97910(16, 5, a9);
      v30 = v29;
      if ( v29 )
      {
        sub_B971C0(v29, (_DWORD)v12, 29, a9, (unsigned int)&v39, 5, 0, 0);
        *(_WORD *)(v30 + 2) = a2;
        *(_DWORD *)(v30 + 4) = a6;
      }
      return sub_B0FB80(v30, a9, v28);
    }
  }
  return result;
}
