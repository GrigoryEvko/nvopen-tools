// Function: sub_32080C0
// Address: 0x32080c0
//
__int64 __fastcall sub_32080C0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  unsigned __int8 *v3; // r12
  __int16 v4; // ax
  __int16 v5; // bx
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  int v8; // r15d
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r14d
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int8 v18; // dl
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // r13
  __int64 *v23; // rbx
  __int64 v24; // r8
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned __int64 v27; // rsi
  __int64 v28; // r12
  unsigned __int8 v29; // al
  __int64 v30; // [rsp+0h] [rbp-1C0h]
  __int64 v31; // [rsp+8h] [rbp-1B8h]
  __int64 v33; // [rsp+28h] [rbp-198h]
  _QWORD *v34; // [rsp+30h] [rbp-190h]
  int v35; // [rsp+38h] [rbp-188h]
  __int16 v36; // [rsp+3Eh] [rbp-182h]
  int v37; // [rsp+4Ch] [rbp-174h]
  unsigned __int64 v38; // [rsp+50h] [rbp-170h] BYREF
  unsigned int v39; // [rsp+58h] [rbp-168h]
  unsigned __int64 v40; // [rsp+60h] [rbp-160h] BYREF
  unsigned __int64 v41; // [rsp+68h] [rbp-158h]
  unsigned int v42; // [rsp+70h] [rbp-150h] BYREF
  char v43; // [rsp+74h] [rbp-14Ch]
  __int64 v44; // [rsp+78h] [rbp-148h]
  __int64 v45; // [rsp+80h] [rbp-140h]
  _WORD v46[3]; // [rsp+90h] [rbp-130h] BYREF
  int v47; // [rsp+96h] [rbp-12Ah]
  unsigned __int64 v48; // [rsp+A0h] [rbp-120h]
  unsigned __int64 v49; // [rsp+A8h] [rbp-118h]
  __int64 v50; // [rsp+B0h] [rbp-110h]
  __int64 v51; // [rsp+B8h] [rbp-108h]
  int v52; // [rsp+C0h] [rbp-100h]

  v2 = a1;
  v3 = (unsigned __int8 *)a2;
  v4 = sub_31F58C0(a2);
  v37 = 0;
  v34 = a1 + 79;
  v36 = v4;
  v33 = a2 - 16;
  v35 = *(_DWORD *)(a2 + 20) & 4;
  if ( v35 )
  {
    LOBYTE(v4) = v4 | 0x80;
    v5 = 0;
    v36 = v4;
    goto LABEL_3;
  }
  sub_3702BF0(v46);
  sub_3702E30(v46, 0);
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a2 - 32);
  else
    v16 = v33 - 8LL * ((v15 >> 2) & 0xF);
  v17 = *(_QWORD *)(v16 + 32);
  if ( v17 )
  {
    v18 = *(_BYTE *)(v17 - 16);
    if ( (v18 & 2) != 0 )
    {
      v19 = *(_QWORD *)(v17 - 32);
      v20 = *(unsigned int *)(v17 - 24);
    }
    else
    {
      v19 = v17 - 16 - 8LL * ((v18 >> 2) & 0xF);
      v20 = (*(_WORD *)(v17 - 16) >> 6) & 0xF;
    }
    v21 = v19 + 8 * v20;
    if ( v21 != v19 )
    {
      v22 = (__int64 *)v19;
      v23 = (__int64 *)v21;
      while ( 1 )
      {
        v28 = *v22;
        if ( *v22 )
        {
          if ( *(_BYTE *)v28 == 11 )
            break;
        }
LABEL_27:
        if ( v23 == ++v22 )
        {
          v2 = a1;
          v3 = (unsigned __int8 *)a2;
          v5 = v35;
          goto LABEL_36;
        }
      }
      v29 = *(_BYTE *)(v28 - 16);
      if ( (v29 & 2) != 0 )
      {
        v24 = **(_QWORD **)(v28 - 32);
        if ( !v24 )
          goto LABEL_32;
      }
      else
      {
        v24 = *(_QWORD *)(v28 - 16 - 8LL * ((v29 >> 2) & 0xF));
        if ( !v24 )
        {
LABEL_32:
          v26 = *(_DWORD *)(v28 + 24);
          v25 = 0;
          v39 = v26;
          if ( v26 > 0x40 )
            goto LABEL_33;
          goto LABEL_23;
        }
      }
      v24 = sub_B91420(v24);
      v26 = *(_DWORD *)(v28 + 24);
      v39 = v26;
      if ( v26 > 0x40 )
      {
LABEL_33:
        v30 = v25;
        v31 = v24;
        sub_C43780((__int64)&v38, (const void **)(v28 + 16));
        v26 = v39;
        v27 = v38;
        v25 = v30;
        v24 = v31;
        goto LABEL_24;
      }
LABEL_23:
      v27 = *(_QWORD *)(v28 + 16);
LABEL_24:
      v41 = v27;
      v40 = 201986;
      v42 = v26;
      v43 = 1;
      v44 = v24;
      v45 = v25;
      sub_3703CD0(v46, &v40);
      LOWORD(v35) = v35 + 1;
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      goto LABEL_27;
    }
  }
  v5 = 0;
LABEL_36:
  v37 = sub_37083C0(v34, v46);
  sub_3702CE0(v46);
LABEL_3:
  sub_3205740((__int64)&v40, (__int64)v2, v3);
  v6 = *(v3 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *((_QWORD *)v3 - 4);
  else
    v7 = v33 - 8LL * ((v6 >> 2) & 0xF);
  v8 = sub_3206530((__int64)v2, *(unsigned __int8 **)(v7 + 24), 0);
  v9 = *(v3 - 16);
  if ( (v9 & 2) != 0 )
  {
    v10 = *(_QWORD *)(*((_QWORD *)v3 - 4) + 56LL);
    if ( v10 )
    {
LABEL_7:
      v10 = sub_B91420(v10);
      goto LABEL_8;
    }
  }
  else
  {
    v10 = *(_QWORD *)(v33 - 8LL * ((v9 >> 2) & 0xF) + 56);
    if ( v10 )
      goto LABEL_7;
  }
  v11 = 0;
LABEL_8:
  v50 = v10;
  v46[0] = 5383;
  v46[1] = v5;
  v46[2] = v36;
  v51 = v11;
  v47 = v37;
  v52 = v8;
  v48 = v40;
  v49 = v41;
  v12 = sub_370A430(v2 + 81, v46);
  v13 = sub_3707F80(v34, v12);
  sub_31FDA50(v2, (__int64)v3, v13);
  if ( (unsigned int *)v40 != &v42 )
    j_j___libc_free_0(v40);
  return v13;
}
