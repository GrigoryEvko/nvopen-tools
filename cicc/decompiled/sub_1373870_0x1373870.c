// Function: sub_1373870
// Address: 0x1373870
//
void __fastcall sub_1373870(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  __int64 v4; // rdx
  __int64 v5; // r10
  unsigned int *v6; // r9
  __int64 v7; // rax
  _DWORD *v8; // rax
  __int64 v9; // r10
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rbx
  unsigned int v13; // r15d
  __int64 v14; // rax
  _DWORD *v15; // rdi
  __int64 v16; // r11
  __int64 v17; // rax
  unsigned __int64 *v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  bool v22; // cf
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 *v25; // r9
  unsigned __int64 *v26; // rdx
  bool v27; // al
  bool v28; // al
  __int64 *v29; // [rsp+10h] [rbp-D0h]
  __int64 *v30; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v31; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v32; // [rsp+18h] [rbp-C8h]
  __int64 v33; // [rsp+20h] [rbp-C0h]
  __int64 v34; // [rsp+20h] [rbp-C0h]
  _BYTE *v35; // [rsp+38h] [rbp-A8h]
  int v36; // [rsp+48h] [rbp-98h] BYREF
  int v37; // [rsp+4Ch] [rbp-94h] BYREF
  _BYTE *v38; // [rsp+50h] [rbp-90h] BYREF
  __int64 v39; // [rsp+58h] [rbp-88h]
  _BYTE v40[64]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-40h]
  char v42; // [rsp+A8h] [rbp-38h]

  v3 = 0;
  v4 = *(unsigned int *)(a2 + 12);
  v38 = v40;
  v39 = 0x400000000LL;
  v41 = 0;
  v42 = 0;
  if ( (_DWORD)v4 )
  {
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(a2 + 96);
        v6 = (unsigned int *)(v5 + 4LL * v3);
        v7 = 0;
        if ( (_DWORD)v4 != 1 )
        {
          v8 = sub_13706F0(*(_DWORD **)(a2 + 96), v5 + 4 * v4, (_DWORD *)(v5 + 4LL * v3));
          v7 = 8 * (((__int64)v8 - v9) >> 2);
        }
        v10 = *(_QWORD *)(*(_QWORD *)(a2 + 128) + v7);
        if ( v10 )
          break;
        v4 = *(unsigned int *)(a2 + 12);
        if ( (unsigned int)v4 <= ++v3 )
          goto LABEL_8;
      }
      ++v3;
      sub_1370BE0((__int64)&v38, v6, v10, 0);
      v4 = *(unsigned int *)(a2 + 12);
    }
    while ( (unsigned int)v4 > v3 );
  }
LABEL_8:
  sub_1372DF0((__int64)&v38);
  v11 = (unsigned __int64)v38;
  v35 = &v38[16 * (unsigned int)v39];
  if ( v35 == v38 )
    goto LABEL_23;
  v12 = -1;
  v13 = v41;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v11 + 8);
      v20 = v13;
      v13 -= v19;
      sub_16AF710(&v37, (unsigned int)v19, v20);
      v36 = v37;
      v21 = sub_16AF780(&v36, v12);
      v22 = v12 < v21;
      v12 -= v21;
      v23 = v21;
      if ( v22 )
        v12 = 0;
      v24 = *(_QWORD *)(a1 + 64) + 24LL * *(unsigned int *)(v11 + 4);
      v25 = *(__int64 **)(v24 + 8);
      if ( !v25 )
        goto LABEL_20;
      v14 = *((unsigned int *)v25 + 3);
      v15 = (_DWORD *)v25[12];
      if ( (unsigned int)v14 > 1 )
        break;
      if ( *(_DWORD *)v24 == *v15 )
        goto LABEL_12;
LABEL_20:
      v26 = (unsigned __int64 *)(v24 + 16);
LABEL_21:
      *v26 = v23;
      v11 += 16LL;
      if ( (_BYTE *)v11 == v35 )
        goto LABEL_22;
    }
    v29 = *(__int64 **)(v24 + 8);
    v31 = v23;
    v33 = *(_QWORD *)(a1 + 64) + 24LL * *(unsigned int *)(v11 + 4);
    v27 = sub_1369030(v15, &v15[v14], (_DWORD *)v24);
    v24 = v33;
    v23 = v31;
    v25 = v29;
    if ( !v27 )
    {
      v26 = (unsigned __int64 *)(v33 + 16);
      goto LABEL_21;
    }
LABEL_12:
    if ( !*((_BYTE *)v25 + 8) )
      goto LABEL_20;
    v16 = *v25;
    if ( !*v25 )
      goto LABEL_15;
    v17 = *(unsigned int *)(v16 + 12);
    if ( (unsigned int)v17 <= 1
      || (v30 = v25,
          v32 = v23,
          v34 = *v25,
          v28 = sub_1369030(*(_DWORD **)(v16 + 96), (_DWORD *)(*(_QWORD *)(v16 + 96) + 4 * v17), (_DWORD *)v24),
          v23 = v32,
          v25 = v30,
          !v28)
      || (v18 = (unsigned __int64 *)(v34 + 152), !*(_BYTE *)(v34 + 8)) )
    {
LABEL_15:
      v18 = (unsigned __int64 *)(v25 + 19);
    }
    *v18 = v23;
    v11 += 16LL;
  }
  while ( (_BYTE *)v11 != v35 );
LABEL_22:
  v11 = (unsigned __int64)v38;
LABEL_23:
  if ( (_BYTE *)v11 != v40 )
    _libc_free(v11);
}
