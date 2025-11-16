// Function: sub_3968710
// Address: 0x3968710
//
__int64 __fastcall sub_3968710(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  _BYTE *v4; // r12
  unsigned __int64 v5; // r9
  int v6; // ebx
  _QWORD *v7; // r10
  unsigned int v8; // edx
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // r12
  __int64 v14; // rax
  int *v15; // rax
  int v16; // r12d
  int v17; // r13d
  __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v21; // rax
  int v23; // [rsp+14h] [rbp-ACh]
  int v25; // [rsp+28h] [rbp-98h]
  float v26; // [rsp+2Ch] [rbp-94h]
  __int64 v27; // [rsp+30h] [rbp-90h] BYREF
  __int64 v28; // [rsp+38h] [rbp-88h] BYREF
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 v30; // [rsp+48h] [rbp-78h] BYREF
  _BYTE *v31; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v32; // [rsp+58h] [rbp-68h]
  _BYTE *v33; // [rsp+60h] [rbp-60h]
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+78h] [rbp-48h]
  __int64 v36; // [rsp+80h] [rbp-40h]
  __int64 v37; // [rsp+88h] [rbp-38h]

  v27 = a3;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  sub_1292090((__int64)&v31, 0, &v27);
  v4 = v32;
  v5 = v35;
  if ( v32 == v31 )
  {
LABEL_31:
    v16 = 0;
    goto LABEL_32;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v11 = *((_QWORD *)v4 - 1);
      v4 -= 8;
      v32 = v4;
      v28 = v11;
      if ( a2 != v11 )
      {
        if ( !(_DWORD)v37 )
        {
          ++v34;
LABEL_8:
          sub_13B3D40((__int64)&v34, 2 * v37);
LABEL_9:
          sub_1898220((__int64)&v34, &v28, &v30);
          v7 = (_QWORD *)v30;
          v11 = v28;
          v12 = v36 + 1;
          goto LABEL_19;
        }
        v6 = 1;
        v7 = 0;
        v8 = (v37 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v9 = (_QWORD *)(v5 + 8LL * v8);
        v10 = *v9;
        if ( v11 != *v9 )
          break;
      }
LABEL_4:
      if ( v31 == v4 )
        goto LABEL_31;
    }
    while ( v10 != -8 )
    {
      if ( v10 != -16 || v7 )
        v9 = v7;
      v8 = (v37 - 1) & (v6 + v8);
      v10 = *(_QWORD *)(v5 + 8LL * v8);
      if ( v11 == v10 )
        goto LABEL_4;
      ++v6;
      v7 = v9;
      v9 = (_QWORD *)(v5 + 8LL * v8);
    }
    if ( !v7 )
      v7 = v9;
    ++v34;
    v12 = v36 + 1;
    if ( 4 * ((int)v36 + 1) >= (unsigned int)(3 * v37) )
      goto LABEL_8;
    if ( (int)v37 - HIDWORD(v36) - v12 <= (unsigned int)v37 >> 3 )
    {
      sub_13B3D40((__int64)&v34, v37);
      goto LABEL_9;
    }
LABEL_19:
    LODWORD(v36) = v12;
    if ( *v7 != -8 )
      --HIDWORD(v36);
    *v7 = v11;
    v13 = *(_QWORD *)(a1 + 56);
    v26 = *(float *)&dword_5056060;
    v29 = v28;
    v14 = (unsigned __int8)sub_39538E0(v13 + 112, &v29, &v30)
        ? v30
        : *(_QWORD *)(v13 + 120) + 16LL * *(unsigned int *)(v13 + 136);
    v15 = *(int **)(v14 + 8);
    v16 = v15[4];
    v17 = *v15;
    v25 = v15[5];
    v23 = v15[1];
    LOBYTE(v16) = sub_1642F90(*a4, 1) ? (float)v23 > (float)((float)v25 * v26) : (float)v17 > (float)((float)v16 * v26);
    if ( (_BYTE)v16 )
      break;
    v4 = v32;
    v18 = *(_QWORD *)(v28 + 8);
    if ( v18 )
    {
      while ( 1 )
      {
        v19 = sub_1648700(v18);
        if ( (unsigned __int8)(*((_BYTE *)v19 + 16) - 25) <= 9u )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_29;
      }
LABEL_36:
      v21 = v19[5];
      v30 = v21;
      if ( v4 == v33 )
      {
        sub_1292090((__int64)&v31, v4, &v30);
        v4 = v32;
      }
      else
      {
        if ( v4 )
        {
          *(_QWORD *)v4 = v21;
          v4 = v32;
        }
        v4 += 8;
        v32 = v4;
      }
      while ( 1 )
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          break;
        v19 = sub_1648700(v18);
        if ( (unsigned __int8)(*((_BYTE *)v19 + 16) - 25) <= 9u )
          goto LABEL_36;
      }
      v5 = v35;
    }
    else
    {
LABEL_29:
      v5 = v35;
    }
    if ( v31 == v4 )
      goto LABEL_31;
  }
  v5 = v35;
LABEL_32:
  j___libc_free_0(v5);
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  return (unsigned int)v16;
}
