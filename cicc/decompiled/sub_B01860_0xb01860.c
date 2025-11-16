// Function: sub_B01860
// Address: 0xb01860
//
_QWORD *__fastcall sub_B01860(
        __int64 *a1,
        int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        char a6,
        unsigned int a7,
        char a8)
{
  int v8; // r10d
  __int64 v9; // r11
  unsigned int v12; // ebx
  __int64 v13; // r15
  int v14; // eax
  int v15; // r9d
  int v16; // r9d
  unsigned int i; // r8d
  __int64 v18; // rcx
  unsigned int v19; // r8d
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdi
  char v24; // r9
  __int64 v25; // r12
  _BYTE *v27; // rax
  unsigned __int8 v28; // al
  __int64 v29; // rax
  __int64 v30; // rdi
  _QWORD **v31; // [rsp+0h] [rbp-A0h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  int v33; // [rsp+10h] [rbp-90h]
  unsigned int v34; // [rsp+14h] [rbp-8Ch]
  _QWORD *v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+20h] [rbp-80h]
  int v38; // [rsp+28h] [rbp-78h]
  int v39; // [rsp+2Ch] [rbp-74h]
  int v40; // [rsp+2Ch] [rbp-74h]
  __int64 v41; // [rsp+40h] [rbp-60h]
  int v42; // [rsp+40h] [rbp-60h]
  __int64 v44; // [rsp+48h] [rbp-58h]
  __int64 *v45; // [rsp+50h] [rbp-50h] BYREF
  __int64 v46; // [rsp+58h] [rbp-48h] BYREF
  __int64 v47; // [rsp+60h] [rbp-40h] BYREF
  __int64 v48[7]; // [rsp+68h] [rbp-38h] BYREF

  v8 = a2;
  v9 = a4;
  v12 = a3;
  if ( a3 >= 0x10000 )
    v12 = 0;
  if ( a7 )
    goto LABEL_12;
  v13 = *a1;
  v45 = (__int64 *)__PAIR64__(v12, a2);
  v46 = a4;
  v47 = a5;
  LOBYTE(v48[0]) = a6;
  v39 = *(_DWORD *)(v13 + 720);
  v41 = *(_QWORD *)(v13 + 704);
  if ( v39 )
  {
    v14 = sub_AF71E0((int *)&v45, (int *)&v45 + 1, &v46, &v47, (__int8 *)v48);
    v15 = v39;
    v8 = a2;
    v40 = 1;
    v9 = a4;
    v16 = v15 - 1;
    for ( i = v16 & v14; ; i = v16 & v19 )
    {
      v18 = *(_QWORD *)(v41 + 8LL * i);
      if ( v18 == -4096 )
        goto LABEL_11;
      if ( v18 != -8192 && v45 == (__int64 *)__PAIR64__(*(unsigned __int16 *)(v18 + 2), *(_DWORD *)(v18 + 4)) )
      {
        v31 = (_QWORD **)(v41 + 8LL * i);
        v32 = v9;
        v33 = v8;
        v34 = i;
        v38 = v16;
        v36 = *v31;
        v37 = v18 - 16;
        v27 = sub_A17150((_BYTE *)(v18 - 16));
        v16 = v38;
        i = v34;
        v8 = v33;
        v9 = v32;
        if ( v46 == *(_QWORD *)v27 )
          break;
      }
LABEL_10:
      v19 = v40 + i;
      ++v40;
    }
    v28 = *((_BYTE *)v36 - 16);
    if ( (v28 & 2) != 0 )
    {
      v29 = 0;
      if ( *((_DWORD *)v36 - 6) != 2 )
      {
LABEL_23:
        if ( v47 == v29 && LOBYTE(v48[0]) == (unsigned __int8)BYTE1(*v36) >> 7 )
        {
          if ( v31 == (_QWORD **)(*(_QWORD *)(v13 + 704) + 8LL * *(unsigned int *)(v13 + 720)) )
            goto LABEL_11;
          return v36;
        }
        goto LABEL_10;
      }
      v30 = *(v36 - 4);
    }
    else
    {
      if ( ((*((_WORD *)v36 - 8) >> 6) & 0xF) != 2 )
      {
        v29 = 0;
        goto LABEL_23;
      }
      v30 = v37 - 8LL * ((v28 >> 2) & 0xF);
    }
    v29 = *(_QWORD *)(v30 + 8);
    goto LABEL_23;
  }
LABEL_11:
  if ( !a8 )
    return 0;
LABEL_12:
  v47 = v9;
  v20 = 1;
  v45 = &v47;
  v46 = 0x200000001LL;
  if ( a5 )
  {
    v48[0] = a5;
    v20 = 2;
    LODWORD(v46) = 2;
  }
  v42 = v8;
  v21 = *a1 + 696;
  v22 = sub_B97910(16, v20, a7);
  v23 = v22;
  if ( v22 )
  {
    v24 = a6;
    v44 = v22;
    sub_AF1680(v22, (int)a1, a7, v42, v12, v24, (__int64)&v47, v20);
    v23 = v44;
  }
  v25 = sub_B01630(v23, a7, v21);
  if ( v45 != &v47 )
    _libc_free(v45, a7);
  return (_QWORD *)v25;
}
