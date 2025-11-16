// Function: sub_269F150
// Address: 0x269f150
//
_BOOL8 __fastcall sub_269F150(__int64 a1, __int64 a2)
{
  int v3; // ebx
  __int64 v4; // r14
  _BOOL8 result; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // r13
  unsigned __int8 v9; // al
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned __int8 *v12; // rdi
  int v13; // eax
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // r9
  __int64 (__fastcall *v22)(__int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rsi
  int v26; // ecx
  unsigned int v27; // eax
  __int64 v28; // rdi
  int v29; // r8d
  __int64 v30; // rcx
  int v31; // edi
  int v32; // edi
  unsigned int v33; // edx
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  char v39; // al
  char v40; // dl
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-D8h]
  char v44; // [rsp+8h] [rbp-D8h]
  char v45; // [rsp+8h] [rbp-D8h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+28h] [rbp-B8h] BYREF
  unsigned __int64 v48; // [rsp+30h] [rbp-B0h]
  __int64 v49[3]; // [rsp+38h] [rbp-A8h] BYREF
  char v50; // [rsp+54h] [rbp-8Ch]
  char v51[16]; // [rsp+58h] [rbp-88h] BYREF
  char v52[8]; // [rsp+68h] [rbp-78h] BYREF
  unsigned __int64 v53; // [rsp+70h] [rbp-70h]
  char v54; // [rsp+84h] [rbp-5Ch]
  char v55[88]; // [rsp+88h] [rbp-58h] BYREF

  v3 = *(_DWORD *)(a1 + 144);
  if ( !v3 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  v4 = *(_QWORD *)(a2 + 208);
  result = 1;
  v7 = *(_QWORD *)(v4 + 32432);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v8 = *(_QWORD *)(v8 + 24);
    v9 = *(_BYTE *)v8;
    if ( *(_BYTE *)v8 )
    {
      if ( v9 == 22 )
      {
        v8 = *(_QWORD *)(v8 + 24);
      }
      else if ( v9 <= 0x1Cu )
      {
        v8 = 0;
      }
      else
      {
        v10 = sub_B43CB0(v8);
        v3 = *(_DWORD *)(a1 + 144);
        v7 = *(_QWORD *)(v4 + 32432);
        v8 = v10;
      }
    }
    v11 = *(_QWORD *)(v7 + 16);
    v46 = a1 + 104;
    if ( !v11 )
    {
LABEL_23:
      sub_2673100(a1, a2);
      return *(_DWORD *)(a1 + 144) == v3;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *(unsigned __int8 **)(v11 + 24);
        v13 = *v12;
        if ( (unsigned __int8)v13 <= 0x1Cu )
          goto LABEL_11;
        v14 = (unsigned int)(v13 - 34);
        if ( (unsigned __int8)v14 > 0x33u )
          goto LABEL_11;
        v15 = 0x8000000000041LL;
        if ( !_bittest64(&v15, v14) )
          goto LABEL_11;
        v47 = *(_QWORD *)(v11 + 24);
        if ( v8 != sub_B491C0((__int64)v12) )
          goto LABEL_11;
        if ( *(_DWORD *)(a1 + 120) )
        {
          v24 = *(_DWORD *)(a1 + 128);
          v19 = v47;
          v25 = *(_QWORD *)(a1 + 112);
          if ( !v24 )
            goto LABEL_11;
          v26 = v24 - 1;
          v27 = (v24 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v28 = *(_QWORD *)(v25 + 8LL * v27);
          if ( v28 != v47 )
          {
            v29 = 1;
            while ( v28 != -4096 )
            {
              v27 = v26 & (v29 + v27);
              v28 = *(_QWORD *)(v25 + 8LL * v27);
              if ( v47 == v28 )
                goto LABEL_18;
              ++v29;
            }
            goto LABEL_11;
          }
        }
        else
        {
          v16 = *(_QWORD **)(a1 + 136);
          v17 = (__int64)&v16[*(unsigned int *)(a1 + 144)];
          v18 = sub_266E350(v16, v17, &v47);
          v19 = v47;
          if ( (_QWORD *)v17 == v18 )
            goto LABEL_11;
        }
LABEL_18:
        if ( **(_BYTE **)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)) != 17 )
          goto LABEL_22;
        v49[0] = 0;
        v48 = v8 & 0xFFFFFFFFFFFFFFFCLL;
        nullsub_1518();
        v20 = sub_269E5E0(a2, v48, v49[0], a1, 0, 0, 1);
        if ( !v20 )
          goto LABEL_22;
        v22 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v20 + 112LL);
        v23 = *(_QWORD *)(v47 + 40);
        if ( v22 == sub_26725F0 )
          break;
        v39 = ((__int64 (__fastcall *)(__int64, __int64))v22)(v20, v23);
LABEL_38:
        if ( !v39 )
          goto LABEL_22;
LABEL_11:
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          goto LABEL_23;
      }
      if ( *(_BYTE *)(v20 + 97) )
      {
        v30 = *(_QWORD *)(v20 + 232);
        v31 = *(_DWORD *)(v20 + 248);
        if ( !v31 )
          goto LABEL_11;
        v32 = v31 - 1;
        v33 = v32 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v34 = v30 + ((unsigned __int64)v33 << 7);
        v35 = *(_QWORD *)v34;
        if ( v23 != *(_QWORD *)v34 )
        {
          v41 = 1;
          while ( v35 != -4096 )
          {
            v21 = (unsigned int)(v41 + 1);
            v42 = v32 & (v33 + v41);
            v33 = v42;
            v34 = v30 + (v42 << 7);
            v35 = *(_QWORD *)v34;
            if ( v23 == *(_QWORD *)v34 )
              goto LABEL_34;
            v41 = v21;
          }
          goto LABEL_11;
        }
LABEL_34:
        v43 = v34;
        LODWORD(v48) = *(_DWORD *)(v34 + 8);
        sub_C8CD80((__int64)v49, (__int64)v51, v34 + 16, v30, v35, v21);
        sub_C8CD80((__int64)v52, (__int64)v55, v43 + 64, v36, v37, v38);
        v39 = v48;
        v40 = v50;
        if ( !v54 )
        {
          v44 = v48;
          _libc_free(v53);
          v40 = v50;
          v39 = v44;
        }
        if ( !v40 )
        {
          v45 = v39;
          _libc_free(v49[1]);
          v39 = v45;
        }
        goto LABEL_38;
      }
LABEL_22:
      sub_266F010(v46, &v47);
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        goto LABEL_23;
    }
  }
  return result;
}
