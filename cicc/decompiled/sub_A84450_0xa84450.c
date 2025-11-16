// Function: sub_A84450
// Address: 0xa84450
//
__int64 __fastcall sub_A84450(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned int v19; // r13d
  signed __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  signed __int64 v30; // r13
  int v31; // ebx
  int v32; // r12d
  signed __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rax
  void *v36; // rdi
  __int64 v37; // [rsp-150h] [rbp-150h]
  signed __int64 v38; // [rsp-148h] [rbp-148h]
  __int64 v39; // [rsp-128h] [rbp-128h]
  __int64 v40; // [rsp-120h] [rbp-120h]
  unsigned int v41; // [rsp-114h] [rbp-114h]
  __int64 v42; // [rsp-108h] [rbp-108h]
  __int64 v43; // [rsp-108h] [rbp-108h]
  void *v44; // [rsp-100h] [rbp-100h]
  __int64 v45; // [rsp-100h] [rbp-100h]
  __int64 v46; // [rsp-100h] [rbp-100h]
  __int64 v47; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v48; // [rsp-F0h] [rbp-F0h]
  __int64 v49; // [rsp-E8h] [rbp-E8h]
  __int16 v50; // [rsp-D8h] [rbp-D8h]
  _QWORD *v51; // [rsp-C8h] [rbp-C8h]
  __int64 v52; // [rsp-C0h] [rbp-C0h]
  _QWORD v53[6]; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v54; // [rsp-88h] [rbp-88h]
  __int64 v55; // [rsp-80h] [rbp-80h]
  void **v56; // [rsp-78h] [rbp-78h]
  _QWORD *v57; // [rsp-70h] [rbp-70h]
  __int64 v58; // [rsp-68h] [rbp-68h]
  int v59; // [rsp-60h] [rbp-60h]
  __int16 v60; // [rsp-5Ch] [rbp-5Ch]
  char v61; // [rsp-5Ah] [rbp-5Ah]
  __int64 v62; // [rsp-58h] [rbp-58h]
  __int64 v63; // [rsp-50h] [rbp-50h]
  void *v64; // [rsp-48h] [rbp-48h] BYREF
  _QWORD v65[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 7) & 0x10) == 0 )
    return 0;
  v2 = a1;
  v3 = sub_BD5D20(a1);
  if ( v4 != 17
    || *(_QWORD *)v3 ^ 0x6F6C672E6D766C6CLL | *(_QWORD *)(v3 + 8) ^ 0x726F74635F6C6162LL
    || *(_BYTE *)(v3 + 16) != 115 )
  {
    v5 = sub_BD5D20(a1);
    if ( v6 != 17
      || *(_QWORD *)v5 ^ 0x6F6C672E6D766C6CLL | *(_QWORD *)(v5 + 8) ^ 0x726F74645F6C6162LL
      || *(_BYTE *)(v5 + 16) != 115 )
    {
      return 0;
    }
  }
  if ( (unsigned __int8)sub_B2FC80(a1) )
    return 0;
  v8 = *(_QWORD *)(a1 + 24);
  result = 0;
  if ( *(_BYTE *)(v8 + 8) == 16 )
  {
    v9 = *(_QWORD *)(v8 + 24);
    if ( *(_BYTE *)(v9 + 8) == 15 && *(_DWORD *)(v9 + 12) == 2 )
    {
      v10 = &v47;
      v11 = sub_BD5C60(a1, a2, v8);
      v58 = 0;
      v60 = 512;
      v51 = v53;
      v52 = 0x200000000LL;
      v56 = &v64;
      v57 = v65;
      v55 = v11;
      v54 = 0;
      v64 = &unk_49DA100;
      v59 = 0;
      v61 = 7;
      v62 = 0;
      v63 = 0;
      v53[4] = 0;
      v53[5] = 0;
      v65[0] = &unk_49DA0B0;
      v12 = sub_BCE3C0(v11, 0);
      v13 = *(_QWORD *)(v9 + 16);
      v14 = *(_QWORD *)(v13 + 8);
      v15 = *(__int64 **)v13;
      v16 = *v15;
      v48 = v14;
      v47 = (__int64)v15;
      v49 = v12;
      v17 = sub_BD0B90(v16, &v47, 3, 0);
      v18 = *(_QWORD *)(v2 - 32);
      v42 = v17;
      v19 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
      v41 = v19;
      v39 = v19;
      if ( v19 )
      {
        v20 = 8LL * v19;
        v44 = (void *)sub_22077B0(v20);
        memset(v44, 0, v20);
        v38 = v20;
        v40 = v20 >> 3;
        v37 = v2;
        v21 = v18;
        v22 = 0;
        do
        {
          if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
            v23 = *(_QWORD *)(v21 - 8);
          else
            v23 = v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF);
          v24 = *(_QWORD *)(v23 + 32 * v22);
          v25 = sub_BCE3C0(v55, 0);
          v26 = sub_AD6530(v25);
          v27 = sub_AD69F0(v24, 1);
          v47 = sub_AD69F0(v24, 0);
          v48 = v27;
          v49 = v26;
          *((_QWORD *)v44 + v22++) = sub_AD24A0(v42, &v47, 3);
        }
        while ( v41 != (_DWORD)v22 );
        v2 = v37;
        LODWORD(v10) = (unsigned int)&v47;
      }
      else
      {
        v40 = 0;
        v38 = 0;
        v44 = 0;
      }
      v28 = sub_BCD420(v42, v39);
      v29 = sub_AD1300(v28, v44, v40);
      v30 = *(_QWORD *)(v29 + 8);
      v31 = v29;
      v32 = *(_BYTE *)(v2 + 32) & 0xF;
      v47 = sub_BD5D20(v2);
      v50 = 261;
      v33 = unk_3F0FAE8;
      v48 = v34;
      v35 = sub_BD2C40(88, unk_3F0FAE8);
      if ( v35 )
      {
        v33 = v30;
        v43 = v35;
        sub_B2FEA0(v35, v30, 0, v32, v31, (_DWORD)v10, 0, 0, 0);
        v35 = v43;
      }
      v36 = v44;
      if ( v44 )
      {
        v33 = v38;
        v45 = v35;
        j_j___libc_free_0(v36, v38);
        v35 = v45;
      }
      v46 = v35;
      nullsub_61(v65);
      v64 = &unk_49DA100;
      nullsub_63(&v64);
      result = v46;
      if ( v51 != v53 )
      {
        _libc_free(v51, v33);
        return v46;
      }
    }
  }
  return result;
}
