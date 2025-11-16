// Function: sub_358FF90
// Address: 0x358ff90
//
__int64 __fastcall sub_358FF90(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rbx
  _QWORD *v7; // r14
  __int64 v8; // rdx
  __int128 v9; // rax
  int *v10; // rax
  size_t v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int128 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned int v22; // edx
  __int64 *v23; // rdi
  __int64 v24; // r10
  __int64 v25; // rax
  __int64 v26; // r12
  __int128 v27; // rax
  int v28; // edi
  int v29; // r11d
  __int64 v30; // [rsp+8h] [rbp-108h]
  __int64 v31; // [rsp+10h] [rbp-100h]
  __int64 v32; // [rsp+10h] [rbp-100h]
  __int64 v33; // [rsp+18h] [rbp-F8h]
  __int64 v34; // [rsp+18h] [rbp-F8h]
  __int128 v35; // [rsp+20h] [rbp-F0h] BYREF
  const char *v36; // [rsp+30h] [rbp-E0h]
  __int64 v37; // [rsp+38h] [rbp-D8h]
  __int64 v38; // [rsp+40h] [rbp-D0h]
  const char *v39; // [rsp+50h] [rbp-C0h]
  __int64 v40; // [rsp+58h] [rbp-B8h]
  char v41; // [rsp+70h] [rbp-A0h]
  char v42; // [rsp+71h] [rbp-9Fh]
  _OWORD v43[2]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-70h]
  void *v45; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v46; // [rsp+B8h] [rbp-58h]
  __int64 v47; // [rsp+C0h] [rbp-50h]
  __int64 v48; // [rsp+C8h] [rbp-48h]
  int v49; // [rsp+D0h] [rbp-40h]
  __int128 *v50; // [rsp+D8h] [rbp-38h]

  v2 = *(unsigned __int8 *)(a1[142] + 184LL);
  if ( (_BYTE)v2 )
  {
    v4 = *a2;
    sub_3585D20((__int64)a1, 0);
    v7 = (_QWORD *)a1[142];
    v45 = (void *)sub_B2D7E0(v4, "sample-profile-suffix-elision-policy", 0x24u);
    v31 = sub_A72240((__int64 *)&v45);
    v33 = v8;
    *(_QWORD *)&v9 = sub_BD5D20(v4);
    v10 = (int *)sub_C16140(v9, v31, v33);
    v12 = sub_26C7880(v7, v10, v11);
    a1[150] = v12;
    v13 = v12;
    if ( !v12 || !*(_QWORD *)(v12 + 56) )
      return 0;
    v14 = *a2;
    if ( byte_4F838D4[0] )
    {
      v30 = a1[149];
      v45 = (void *)sub_B2D7E0(*a2, "sample-profile-suffix-elision-policy", 0x24u);
      v32 = sub_A72240((__int64 *)&v45);
      v34 = v15;
      *(_QWORD *)&v16 = sub_BD5D20(v14);
      v17 = sub_C16140(v16, v32, v34);
      v19 = sub_B2F650(v17, v18);
      v20 = *(unsigned int *)(v30 + 24);
      v21 = *(_QWORD *)(v30 + 8);
      if ( (_DWORD)v20 )
      {
        v22 = (v20 - 1) & (((0xBF58476D1CE4E5B9LL * v19) >> 31) ^ (484763065 * v19));
        v23 = (__int64 *)(v21 + 24LL * v22);
        v24 = *v23;
        if ( v19 == *v23 )
        {
LABEL_8:
          if ( v23 != (__int64 *)(v21 + 24 * v20) && (*(_BYTE *)(v14 + 32) & 0xF) != 1 )
          {
            if ( v23[2] != *(_QWORD *)(v13 + 8) )
              return 0;
            goto LABEL_14;
          }
        }
        else
        {
          v28 = 1;
          while ( v24 != -1 )
          {
            v29 = v28 + 1;
            v22 = (v20 - 1) & (v28 + v22);
            v23 = (__int64 *)(v21 + 24LL * v22);
            v24 = *v23;
            if ( v19 == *v23 )
              goto LABEL_8;
            v28 = v29;
          }
        }
      }
      if ( (unsigned __int8)sub_B2D620(v14, "profile-checksum-mismatch", 0x19u) )
        return 0;
      goto LABEL_14;
    }
    v25 = sub_B92180(v14);
    if ( !v25 )
    {
      v2 = LOBYTE(qword_500BB08[8]);
      if ( !LOBYTE(qword_500BB08[8]) )
      {
        v42 = 1;
        v26 = sub_B2BE50(v14);
        v41 = 3;
        v39 = ": Function profile not used";
        *(_QWORD *)&v27 = sub_BD5D20(v14);
        v43[1] = v27;
        LOWORD(v44) = 1283;
        *(_QWORD *)&v43[0] = "No debug information found in function ";
        v36 = ": Function profile not used";
        *(_QWORD *)&v35 = v43;
        LOWORD(v38) = 770;
        v37 = v40;
        v47 = 0;
        v46 = 0x10000000CLL;
        v48 = 0;
        v49 = 0;
        v45 = &unk_49D9C78;
        v50 = &v35;
        sub_B6EB20(v26, (__int64)&v45);
        return v2;
      }
      return 0;
    }
    if ( !*(_DWORD *)(v25 + 16) )
      return 0;
LABEL_14:
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v2 = sub_358FB10((__int64)a1, a2, (__int64)&v45);
    sub_3589A20((__int64)a1, (__int64)a2);
    sub_C7D6A0(v46, 8LL * (unsigned int)v48, 8);
  }
  return v2;
}
