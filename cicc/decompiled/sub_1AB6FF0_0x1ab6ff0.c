// Function: sub_1AB6FF0
// Address: 0x1ab6ff0
//
__int64 __fastcall sub_1AB6FF0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  __int64 *v5; // rbx
  _BYTE *v6; // r8
  __int64 *v7; // r13
  _QWORD **v8; // r10
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 v11; // rdx
  __int64 *v12; // r9
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // r13
  const char *v17; // rax
  char v18; // bl
  __int64 v19; // rdx
  char v20; // bl
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdi
  unsigned int v27; // esi
  __int64 v28; // rdx
  __int64 v29; // r10
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  int v36; // edx
  int v37; // r11d
  int v38; // edx
  int v39; // ecx
  __int64 v41; // [rsp+20h] [rbp-C0h]
  __int64 v42; // [rsp+28h] [rbp-B8h]
  __int64 v43; // [rsp+28h] [rbp-B8h]
  _QWORD **v44; // [rsp+28h] [rbp-B8h]
  const char *v45; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-A8h]
  _QWORD *v47; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v48; // [rsp+48h] [rbp-98h]
  _BYTE *v49; // [rsp+50h] [rbp-90h]
  __int64 v50[2]; // [rsp+60h] [rbp-80h] BYREF
  _WORD v51[56]; // [rsp+70h] [rbp-70h] BYREF

  v3 = a2;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, a2);
    v5 = *(__int64 **)(a1 + 88);
    v7 = &v5[5 * *(_QWORD *)(a1 + 96)];
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(a1, a2);
      v5 = *(__int64 **)(a1 + 88);
    }
    v6 = v48;
  }
  else
  {
    v5 = *(__int64 **)(a1 + 88);
    v6 = 0;
    v7 = &v5[5 * *(_QWORD *)(a1 + 96)];
  }
  v8 = &v47;
  if ( v5 != v7 )
  {
    while ( 1 )
    {
      v13 = *(unsigned int *)(v3 + 24);
      if ( !(_DWORD)v13 )
        goto LABEL_9;
      v9 = *(_QWORD *)(v3 + 8);
      v10 = (v13 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = v9 + ((unsigned __int64)v10 << 6);
      v12 = *(__int64 **)(v11 + 24);
      if ( v5 == v12 )
      {
LABEL_6:
        if ( v11 == v9 + (v13 << 6) )
          goto LABEL_9;
LABEL_7:
        v5 += 5;
        if ( v7 == v5 )
          break;
      }
      else
      {
        v36 = 1;
        while ( v12 != (__int64 *)-8LL )
        {
          v37 = v36 + 1;
          v10 = (v13 - 1) & (v36 + v10);
          v11 = v9 + ((unsigned __int64)v10 << 6);
          v12 = *(__int64 **)(v11 + 24);
          if ( v12 == v5 )
            goto LABEL_6;
          v36 = v37;
        }
LABEL_9:
        v14 = *v5;
        v50[0] = *v5;
        if ( v6 == v49 )
        {
          v44 = v8;
          sub_1278040((__int64)v8, v6, v50);
          v6 = v48;
          v8 = v44;
          goto LABEL_7;
        }
        if ( v6 )
        {
          *(_QWORD *)v6 = v14;
          v6 = v48;
        }
        v6 += 8;
        v5 += 5;
        v48 = v6;
        if ( v7 == v5 )
          break;
      }
    }
  }
  v15 = v47;
  v16 = sub_1644EA0(
          **(__int64 ***)(*(_QWORD *)(a1 + 24) + 16LL),
          v47,
          (v6 - (_BYTE *)v47) >> 3,
          *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8 != 0);
  v42 = *(_QWORD *)(a1 + 40);
  v17 = sub_1649960(a1);
  v18 = *(_BYTE *)(a1 + 32);
  v45 = v17;
  v46 = v19;
  v20 = v18 & 0xF;
  v51[0] = 261;
  v50[0] = (__int64)&v45;
  v41 = sub_1648B60(120);
  if ( v41 )
  {
    v15 = (_QWORD *)v16;
    sub_15E2490(v41, v16, v20, (__int64)v50, v42);
  }
  if ( (*(_BYTE *)(v41 + 18) & 1) != 0 )
    sub_15E08E0(v41, (__int64)v15);
  v21 = *(_QWORD *)(v41 + 88);
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, (__int64)v15);
    v22 = *(_QWORD *)(a1 + 88);
    v43 = v22 + 40LL * *(_QWORD *)(a1 + 96);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      sub_15E08E0(a1, (__int64)v15);
      v22 = *(_QWORD *)(a1 + 88);
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 88);
    v43 = v22 + 40LL * *(_QWORD *)(a1 + 96);
  }
  if ( v43 != v22 )
  {
    v23 = v3;
    v24 = v22;
    v25 = v23;
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = *(unsigned int *)(v25 + 24);
        if ( !(_DWORD)v30 )
          goto LABEL_25;
        v26 = *(_QWORD *)(v25 + 8);
        v27 = (v30 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v28 = v26 + ((unsigned __int64)v27 << 6);
        v29 = *(_QWORD *)(v28 + 24);
        if ( v24 == v29 )
          break;
        v38 = 1;
        while ( v29 != -8 )
        {
          v39 = v38 + 1;
          v27 = (v30 - 1) & (v38 + v27);
          v28 = v26 + ((unsigned __int64)v27 << 6);
          v29 = *(_QWORD *)(v28 + 24);
          if ( v24 == v29 )
            goto LABEL_22;
          v38 = v39;
        }
LABEL_25:
        v45 = sub_1649960(v24);
        v51[0] = 261;
        v46 = v31;
        v50[0] = (__int64)&v45;
        sub_164B780(v21, v50);
        v32 = sub_1AB4240(v25, v24);
        v33 = v32[2];
        if ( v21 == v33 )
          goto LABEL_38;
        if ( v33 != -8 && v33 != 0 && v33 != -16 )
          sub_1649B30(v32);
        v32[2] = v21;
        if ( v21 == -8 || v21 == 0 || v21 == -16 )
        {
LABEL_38:
          v21 += 40;
          goto LABEL_23;
        }
        sub_164C220((__int64)v32);
        v21 += 40;
        v24 += 40;
        if ( v43 == v24 )
        {
LABEL_32:
          v3 = v25;
          goto LABEL_33;
        }
      }
LABEL_22:
      if ( v28 == v26 + (v30 << 6) )
        goto LABEL_25;
LABEL_23:
      v24 += 40;
      if ( v43 == v24 )
        goto LABEL_32;
    }
  }
LABEL_33:
  v50[0] = (__int64)v51;
  v50[1] = 0x800000000LL;
  v34 = sub_1626D20(a1);
  sub_1AB5B80((_QWORD *)v41, (_BYTE *)a1, v3, v34 != 0, (__int64)v50, byte_3F871B3, a3, 0, 0);
  if ( (_WORD *)v50[0] != v51 )
    _libc_free(v50[0]);
  if ( v47 )
    j_j___libc_free_0(v47, v49 - (_BYTE *)v47);
  return v41;
}
