// Function: sub_DD8EB0
// Address: 0xdd8eb0
//
__int64 *__fastcall sub_DD8EB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r14
  char v9; // bl
  __int64 *v10; // rbx
  __int64 v11; // r8
  char v12; // r15
  __int64 v13; // r13
  __int64 v14; // r9
  __int64 v15; // r9
  _QWORD *v16; // rcx
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  unsigned __int8 *v20; // r9
  _QWORD *v21; // rdx
  _BYTE *v22; // rdi
  __int64 v24; // rax
  _QWORD *v25; // r15
  __int64 *v26; // rax
  __int64 v27; // r8
  _QWORD *v28; // r9
  __int64 *v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // rax
  int v33; // eax
  int v34; // ebx
  char v35; // al
  unsigned __int8 *v36; // [rsp+0h] [rbp-C0h]
  int v37; // [rsp+Ch] [rbp-B4h]
  __int64 *v38; // [rsp+20h] [rbp-A0h]
  unsigned int v39; // [rsp+28h] [rbp-98h]
  unsigned int v40; // [rsp+2Ch] [rbp-94h]
  unsigned __int8 *v41; // [rsp+30h] [rbp-90h]
  __int64 v42; // [rsp+30h] [rbp-90h]
  _QWORD *v43; // [rsp+30h] [rbp-90h]
  __int64 *v44; // [rsp+38h] [rbp-88h]
  _QWORD *v45; // [rsp+38h] [rbp-88h]
  _QWORD v46[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v47[2]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v48; // [rsp+60h] [rbp-60h] BYREF
  __int64 v49; // [rsp+68h] [rbp-58h]
  _BYTE v50[80]; // [rsp+70h] [rbp-50h] BYREF

  v38 = sub_DD8400((__int64)a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v4 = sub_D95540((__int64)v38);
  v8 = sub_D97090((__int64)a1, v4);
  v9 = *(_BYTE *)(a2 + 1) >> 1;
  if ( v9 && *(_BYTE *)a2 > 0x1Cu && (unsigned __int8)sub_DD8590(a1, a2, v5, v6, v7) )
  {
    v37 = v9 & 2;
    v33 = (2 * v9) & 4;
    v34 = v9 & 4;
    v39 = (unsigned __int8)v34;
    v40 = v33;
    if ( v34 )
      v40 = v33 | 2;
    else
      v39 = 0;
  }
  else
  {
    v37 = 0;
    v39 = 0;
    v40 = 0;
  }
  v10 = *(__int64 **)a3;
  v11 = *(_QWORD *)(a2 + 8);
  v48 = v50;
  v49 = 0x400000000LL;
  v44 = &v10[*(unsigned int *)(a3 + 8)];
  if ( v44 == v10 )
    return v38;
  v12 = 1;
  v13 = v11;
  do
  {
    v14 = *v10;
    if ( *(_BYTE *)(v13 + 8) == 15 )
    {
      if ( *(_WORD *)(v14 + 24) )
      {
        v32 = sub_DA3860(a1, a2);
        v22 = v48;
        v38 = v32;
        if ( v48 == v50 )
          return v38;
        goto LABEL_30;
      }
      v15 = *(_QWORD *)(v14 + 32);
      v16 = *(_QWORD **)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
        v16 = (_QWORD *)*v16;
      v41 = (unsigned __int8 *)v15;
      v17 = sub_DA2D20((__int64)a1, v8, v13, (unsigned int)v16);
      v19 = (unsigned int)v49;
      v20 = v41;
      if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
      {
        v36 = v41;
        v43 = v17;
        sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 8u, v18, (__int64)v20);
        v19 = (unsigned int)v49;
        v20 = v36;
        v17 = v43;
      }
      *(_QWORD *)&v48[8 * v19] = v17;
      LODWORD(v49) = v49 + 1;
      v13 = sub_BCBAE0(v13, v20, v19);
    }
    else
    {
      v42 = *v10;
      if ( v12 )
        v24 = sub_BB5290(a2);
      else
        v24 = sub_B4DC00(v13, 0);
      v13 = v24;
      v25 = sub_DCAD70(a1, v8, v24);
      v47[0] = sub_DC5140((__int64)a1, v42, v8, 0);
      v46[0] = v47;
      v47[1] = v25;
      v46[1] = 0x200000002LL;
      v26 = sub_DC8BD0(a1, (__int64)v46, v40, 0);
      v28 = v47;
      v29 = v26;
      if ( (_QWORD *)v46[0] != v47 )
        _libc_free(v46[0], v46);
      v30 = (unsigned int)v49;
      v31 = (unsigned int)v49 + 1LL;
      if ( v31 > HIDWORD(v49) )
      {
        sub_C8D5F0((__int64)&v48, v50, v31, 8u, v27, (__int64)v28);
        v30 = (unsigned int)v49;
      }
      *(_QWORD *)&v48[8 * v30] = v29;
      v12 = 0;
      LODWORD(v49) = v49 + 1;
    }
    ++v10;
  }
  while ( v44 != v10 );
  a2 = (unsigned int)v49;
  if ( (_DWORD)v49 )
  {
    v21 = sub_DC7EB0(a1, (__int64)&v48, v40, 0);
    if ( v39 || v37 && (v45 = v21, v35 = sub_DBED40((__int64)a1, (__int64)v21), v21 = v45, v35) )
      v39 = 2;
    a2 = (__int64)v38;
    v38 = sub_DC7ED0(a1, (__int64)v38, (__int64)v21, v39, 0);
  }
  v22 = v48;
  if ( v48 != v50 )
LABEL_30:
    _libc_free(v22, a2);
  return v38;
}
