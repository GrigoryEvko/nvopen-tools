// Function: sub_A89080
// Address: 0xa89080
//
__int64 __fastcall sub_A89080(_QWORD *a1, const char *a2, unsigned int a3)
{
  size_t v4; // rdx
  __int64 v5; // r14
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  unsigned __int64 v17; // r15
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  int v24; // ebx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // r14
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // r8d
  __int64 v36; // r13
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // [rsp+0h] [rbp-170h]
  __int64 v41; // [rsp+30h] [rbp-140h]
  __int64 v42; // [rsp+38h] [rbp-138h]
  __int64 v43; // [rsp+48h] [rbp-128h]
  unsigned int v44; // [rsp+58h] [rbp-118h]
  _BYTE *v45; // [rsp+60h] [rbp-110h] BYREF
  __int64 v46; // [rsp+68h] [rbp-108h]
  _BYTE v47[16]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE v48[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v49; // [rsp+A0h] [rbp-D0h]
  unsigned int *v50[2]; // [rsp+B0h] [rbp-C0h] BYREF
  _BYTE v51[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+E0h] [rbp-90h]
  __int64 v53; // [rsp+E8h] [rbp-88h]
  __int16 v54; // [rsp+F0h] [rbp-80h]
  __int64 v55; // [rsp+F8h] [rbp-78h]
  void **v56; // [rsp+100h] [rbp-70h]
  _QWORD *v57; // [rsp+108h] [rbp-68h]
  __int64 v58; // [rsp+110h] [rbp-60h]
  int v59; // [rsp+118h] [rbp-58h]
  __int16 v60; // [rsp+11Ch] [rbp-54h]
  char v61; // [rsp+11Eh] [rbp-52h]
  __int64 v62; // [rsp+120h] [rbp-50h]
  __int64 v63; // [rsp+128h] [rbp-48h]
  void *v64; // [rsp+130h] [rbp-40h] BYREF
  _QWORD v65[7]; // [rsp+138h] [rbp-38h] BYREF

  v4 = 0;
  v5 = *a1;
  if ( a2 )
    v4 = strlen(a2);
  result = sub_BA8CB0(v5, a2, v4);
  v43 = result;
  v7 = result;
  if ( result )
  {
    v8 = sub_B6E160(*a1, a3, 0, 0);
    v9 = *(_QWORD *)(v7 + 16);
    v42 = v8;
    if ( !v9 )
      return sub_B2E860(v43);
    while ( 1 )
    {
      v10 = v9;
      v9 = *(_QWORD *)(v9 + 8);
      v11 = *(unsigned __int8 **)(v10 + 24);
      if ( *v11 != 85 )
        goto LABEL_6;
      v12 = *((_QWORD *)v11 - 4);
      if ( !v12 || *(_BYTE *)v12 || *((_QWORD *)v11 + 10) != *(_QWORD *)(v12 + 24) || v43 != v12 )
        goto LABEL_6;
      v13 = *((_QWORD *)v11 + 5);
      v14 = sub_AA48A0(v13);
      v61 = 7;
      v55 = v14;
      v56 = &v64;
      v57 = v65;
      v50[0] = (unsigned int *)v51;
      v64 = &unk_49DA100;
      v50[1] = (unsigned int *)0x200000000LL;
      v65[0] = &unk_49DA0B0;
      v58 = 0;
      v59 = 0;
      v60 = 512;
      v62 = 0;
      v63 = 0;
      v52 = 0;
      v53 = 0;
      v54 = 0;
      sub_A88F30((__int64)v50, v13, (__int64)(v11 + 24), 0);
      v46 = 0x200000000LL;
      v16 = *((_QWORD *)v11 + 1);
      v17 = *(_QWORD *)(v42 + 24);
      v45 = v47;
      v18 = **(_QWORD **)(v17 + 16);
      if ( v16 == v18 || (unsigned __int8)sub_B50F30(49, v16, v18, v15) )
        break;
LABEL_14:
      v19 = v45;
      if ( v45 != v47 )
        goto LABEL_15;
LABEL_16:
      nullsub_61(v65);
      v64 = &unk_49DA100;
      nullsub_63(&v64);
      if ( (_BYTE *)v50[0] == v51 )
      {
LABEL_6:
        if ( !v9 )
          goto LABEL_18;
      }
      else
      {
        _libc_free(v50[0], v16);
        if ( !v9 )
        {
LABEL_18:
          result = v43;
          if ( *(_QWORD *)(v43 + 16) )
            return result;
          return sub_B2E860(v43);
        }
      }
    }
    v20 = *v11;
    if ( v20 == 40 )
    {
      v41 = 32LL * (unsigned int)sub_B491D0(v11);
    }
    else
    {
      v41 = 0;
      if ( v20 != 85 )
      {
        if ( v20 != 34 )
          goto LABEL_54;
        v41 = 64;
      }
    }
    if ( (v11[7] & 0x80u) != 0 )
    {
      v21 = sub_BD2BC0(v11);
      v23 = v21 + v22;
      if ( (v11[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v23 >> 4) )
LABEL_54:
          BUG();
      }
      else if ( (unsigned int)((v23 - sub_BD2BC0(v11)) >> 4) )
      {
        if ( (v11[7] & 0x80u) == 0 )
          goto LABEL_54;
        v24 = *(_DWORD *)(sub_BD2BC0(v11) + 8);
        if ( (v11[7] & 0x80u) == 0 )
          BUG();
        v25 = sub_BD2BC0(v11);
        v27 = 32LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v24);
LABEL_34:
        v28 = (32LL * (*((_DWORD *)v11 + 1) & 0x7FFFFFF) - 32 - v41 - v27) >> 5;
        if ( (_DWORD)v28 )
        {
          v29 = (unsigned int)v28;
          v30 = 0;
          v31 = *((_DWORD *)v11 + 1) & 0x7FFFFFF;
          v40 = v9;
          v32 = (unsigned int)v28;
          while ( 1 )
          {
            v36 = *(_QWORD *)&v11[32 * (v30 - v31)];
            if ( *(_DWORD *)(v17 + 12) - 1 <= (unsigned int)v30 )
            {
              ++v30;
            }
            else
            {
              ++v30;
              v16 = *(_QWORD *)(v36 + 8);
              if ( !(unsigned __int8)sub_B50F30(49, v16, *(_QWORD *)(*(_QWORD *)(v17 + 16) + 8 * v30), v29) )
              {
                v9 = v40;
                goto LABEL_14;
              }
              v49 = 257;
              v36 = sub_A7EAA0(v50, 0x31u, v36, *(_QWORD *)(*(_QWORD *)(v17 + 16) + 8 * v30), (__int64)v48, 0, v44, 0);
            }
            v33 = (unsigned int)v46;
            v29 = HIDWORD(v46);
            v34 = (unsigned int)v46 + 1LL;
            if ( v34 > HIDWORD(v46) )
            {
              sub_C8D5F0(&v45, v47, v34, 8);
              v33 = (unsigned int)v46;
            }
            *(_QWORD *)&v45[8 * v33] = v36;
            v35 = v46 + 1;
            LODWORD(v46) = v46 + 1;
            if ( v30 == v32 )
            {
              v9 = v40;
              goto LABEL_45;
            }
            v31 = *((_DWORD *)v11 + 1) & 0x7FFFFFF;
          }
        }
        v35 = v46;
LABEL_45:
        v49 = 257;
        v37 = sub_921880(v50, v17, v42, (int)v45, v35, (__int64)v48, 0);
        *(_WORD *)(v37 + 2) = *((_WORD *)v11 + 1) & 3 | *(_WORD *)(v37 + 2) & 0xFFFC;
        sub_BD6B90(v37, v11);
        v49 = 257;
        v16 = sub_A7EAA0(v50, 0x31u, v37, *((_QWORD *)v11 + 1), (__int64)v48, 0, v44, 0);
        if ( *((_QWORD *)v11 + 2) )
          sub_BD84D0(v11, v16);
        sub_B43D60(v11, v16, v38, v39);
        v19 = v45;
        if ( v45 == v47 )
          goto LABEL_16;
LABEL_15:
        _libc_free(v19, v16);
        goto LABEL_16;
      }
    }
    v27 = 0;
    goto LABEL_34;
  }
  return result;
}
