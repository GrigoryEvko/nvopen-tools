// Function: sub_23FFFB0
// Address: 0x23fffb0
//
__int64 *__fastcall sub_23FFFB0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r12
  int v12; // edx
  __int64 v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r15
  char v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 *v20; // rdx
  _QWORD *v21; // rax
  __int64 *v22; // rdx
  __int64 *v23; // r14
  char v24; // r13
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  int v27; // r9d
  _QWORD *v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // r14
  char v31; // r13
  __int64 v32; // rax
  _QWORD *v33; // [rsp+0h] [rbp-100h]
  _QWORD *i; // [rsp+10h] [rbp-F0h]
  __int64 *v36; // [rsp+18h] [rbp-E8h]
  __int64 j; // [rsp+20h] [rbp-E0h]
  __int64 v38; // [rsp+28h] [rbp-D8h] BYREF
  _BYTE v39[16]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+40h] [rbp-C0h]
  __int64 v41; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+68h] [rbp-98h] BYREF
  unsigned __int64 v43; // [rsp+70h] [rbp-90h]
  __int64 *v44; // [rsp+78h] [rbp-88h]
  __int64 *v45; // [rsp+80h] [rbp-80h]
  __int64 v46; // [rsp+88h] [rbp-78h]
  unsigned __int64 v47; // [rsp+90h] [rbp-70h] BYREF
  _BYTE v48[8]; // [rsp+98h] [rbp-68h] BYREF
  int v49; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v50; // [rsp+A8h] [rbp-58h]
  int *v51; // [rsp+B0h] [rbp-50h]
  int *v52; // [rsp+B8h] [rbp-48h]
  __int64 v53; // [rsp+C0h] [rbp-40h]

  v5 = *(unsigned int *)(a3 + 24);
  v38 = a1;
  v6 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v5 )
    goto LABEL_8;
  v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v8 = (__int64 *)(v6 + 56LL * v7);
  v9 = *v8;
  if ( a1 != *v8 )
  {
    v12 = 1;
    while ( v9 != -4096 )
    {
      v27 = v12 + 1;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 56LL * v7);
      v9 = *v8;
      if ( a1 == *v8 )
        goto LABEL_3;
      v12 = v27;
    }
LABEL_8:
    v46 = 0;
    LODWORD(v42) = 0;
    v43 = 0;
    v44 = &v42;
    v45 = &v42;
    if ( *(_BYTE *)a1 <= 0x1Cu )
    {
      if ( *(_BYTE *)a1 == 22 && (v28 = sub_23FDE00((__int64)&v41, (unsigned __int64 *)&v38), (v30 = v29) != 0) )
      {
        v31 = v28 || v29 == &v42 || a1 < v29[4];
        v32 = sub_22077B0(0x28u);
        *(_QWORD *)(v32 + 32) = v38;
        sub_220F040(v31, v32, v30, &v42);
        ++v46;
        v26 = v43;
        v47 = v38;
        if ( v43 )
        {
LABEL_31:
          v50 = v26;
          v49 = v42;
          v51 = (int *)v44;
          v52 = (int *)v45;
          *(_QWORD *)(v26 + 8) = &v49;
          v43 = 0;
          v53 = v46;
          v44 = &v42;
          v45 = &v42;
          v46 = 0;
LABEL_32:
          sub_23FFD80((__int64)v39, a3, (__int64 *)&v47, (__int64)v48);
          v10 = v40 + 8;
          sub_23FBEA0(v50);
          sub_23FBEA0(v43);
          return (__int64 *)v10;
        }
      }
      else
      {
        v47 = a1;
      }
    }
    else
    {
      if ( !(unsigned __int8)sub_23FAAE0((unsigned __int8 *)a1, a2) )
      {
        v47 = a1;
        v21 = sub_23FDE00((__int64)&v41, &v47);
        v23 = v22;
        if ( v22 )
        {
          v24 = v21 || v22 == &v42 || a1 < v22[4];
          v25 = sub_22077B0(0x28u);
          *(_QWORD *)(v25 + 32) = v47;
          sub_220F040(v24, v25, v23, &v42);
          ++v46;
        }
        v47 = v38;
        if ( v43 )
        {
          v50 = v43;
          v49 = v42;
          v51 = (int *)v44;
          v52 = (int *)v45;
          *(_QWORD *)(v43 + 8) = &v49;
          v43 = 0;
          v53 = v46;
          v44 = &v42;
          v45 = &v42;
          v46 = 0;
        }
        else
        {
          v49 = 0;
          v50 = 0;
          v51 = &v49;
          v52 = &v49;
          v53 = 0;
        }
        goto LABEL_32;
      }
      v13 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v14 = *(_QWORD **)(a1 - 8);
        v33 = &v14[v13];
      }
      else
      {
        v33 = (_QWORD *)a1;
        v14 = (_QWORD *)(a1 - v13 * 8);
      }
      for ( i = v14; v33 != i; i += 4 )
      {
        v15 = sub_23FFFB0(*i, a2, a3);
        v16 = *(_QWORD *)(v15 + 24);
        for ( j = v15 + 8; j != v16; v16 = sub_220EF30(v16) )
        {
          v19 = sub_23FE670(&v41, (__int64)&v42, (unsigned __int64 *)(v16 + 32));
          if ( v20 )
          {
            v17 = v19 || v20 == &v42 || *(_QWORD *)(v16 + 32) < (unsigned __int64)v20[4];
            v36 = v20;
            v18 = sub_22077B0(0x28u);
            *(_QWORD *)(v18 + 32) = *(_QWORD *)(v16 + 32);
            sub_220F040(v17, v18, v36, &v42);
            ++v46;
          }
        }
      }
      v47 = v38;
      v26 = v43;
      if ( v43 )
        goto LABEL_31;
    }
    v49 = 0;
    v50 = 0;
    v51 = &v49;
    v52 = &v49;
    v53 = 0;
    goto LABEL_32;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v6 + 56 * v5) )
    goto LABEL_8;
  return v8 + 1;
}
