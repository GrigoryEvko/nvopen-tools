// Function: sub_12BE2E0
// Address: 0x12be2e0
//
__int64 __fastcall sub_12BE2E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 *v4; // rdi
  __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r10
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r13
  __int64 i; // r14
  __int64 v17; // r15
  __int64 *v18; // r12
  size_t v19; // rbx
  __int64 v20; // rcx
  void *v21; // rdi
  void *v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]
  __int64 v29; // [rsp+30h] [rbp-40h]
  unsigned __int64 v30; // [rsp+38h] [rbp-38h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = (__int64 *)(a1 + 16);
  *((_DWORD *)v4 - 4) = *(_DWORD *)a2;
  *((_DWORD *)v4 - 2) = *(_DWORD *)(a2 + 8);
  *(_QWORD *)(v3 + 16) = v3 + 32;
  sub_12BCB70(v4, *(_BYTE **)(a2 + 16), *(_QWORD *)(a2 + 16) + *(_QWORD *)(a2 + 24));
  *(_QWORD *)(v3 + 48) = v3 + 64;
  sub_12BCB70((__int64 *)(v3 + 48), *(_BYTE **)(v2 + 48), *(_QWORD *)(v2 + 48) + *(_QWORD *)(v2 + 56));
  *(_QWORD *)(v3 + 80) = v3 + 96;
  sub_12BCB70((__int64 *)(v3 + 80), *(_BYTE **)(v2 + 80), *(_QWORD *)(v2 + 80) + *(_QWORD *)(v2 + 88));
  *(_QWORD *)(v3 + 112) = v3 + 128;
  sub_12BCB70((__int64 *)(v3 + 112), *(_BYTE **)(v2 + 112), *(_QWORD *)(v2 + 112) + *(_QWORD *)(v2 + 120));
  *(_QWORD *)(v3 + 144) = v3 + 160;
  sub_12BCB70((__int64 *)(v3 + 144), *(_BYTE **)(v2 + 144), *(_QWORD *)(v2 + 144) + *(_QWORD *)(v2 + 152));
  *(_QWORD *)(v3 + 176) = v3 + 192;
  sub_12BCB70((__int64 *)(v3 + 176), *(_BYTE **)(v2 + 176), *(_QWORD *)(v2 + 176) + *(_QWORD *)(v2 + 184));
  *(_QWORD *)(v3 + 208) = 0;
  *(_QWORD *)(v3 + 216) = 0;
  *(_QWORD *)(v3 + 224) = 0x1000000000LL;
  if ( *(_DWORD *)(a2 + 220) )
  {
    sub_16D1890(v3 + 208, *(unsigned int *)(a2 + 216));
    v8 = *(_QWORD *)(v3 + 208);
    v9 = *(_QWORD *)(a2 + 208);
    v10 = *(unsigned int *)(v3 + 216);
    v11 = 8 * v10 + 8;
    v25 = v8;
    v12 = *(_QWORD *)(a2 + 220);
    v24 = v9;
    *(_QWORD *)(v3 + 220) = v12;
    if ( (_DWORD)v10 )
    {
      v26 = v3;
      v27 = 8LL * (unsigned int)(v10 - 1);
      v13 = v9;
      v14 = 0;
      v15 = v11;
      for ( i = 0; ; i += 8 )
      {
        v17 = *(_QWORD *)(v13 + i);
        v18 = (__int64 *)(v8 + i);
        if ( v17 != -8 )
        {
          if ( v17 )
            break;
        }
        *v18 = v17;
        v15 += 4;
        if ( v27 == i )
        {
LABEL_13:
          v3 = v26;
          v2 = a2;
          goto LABEL_2;
        }
LABEL_6:
        v14 = v26;
        v13 = *(_QWORD *)(a2 + 208);
        v8 = *(_QWORD *)(v26 + 208);
      }
      v19 = *(_QWORD *)v17;
      v28 = *(_QWORD *)v17 + 17LL;
      v30 = *(_QWORD *)v17 + 1LL;
      v6 = malloc(v28, v28, v12, v14, v6, v7);
      if ( !v6 )
      {
        if ( !v28 )
        {
          v23 = malloc(1, 0, v12, v20, 0, v7);
          v6 = 0;
          if ( v23 )
          {
            v21 = (void *)(v23 + 16);
            v6 = v23;
LABEL_11:
            v31 = v6;
            v22 = memcpy(v21, (const void *)(v17 + 16), v19);
            v6 = v31;
            v21 = v22;
LABEL_5:
            *((_BYTE *)v21 + v19) = 0;
            *(_QWORD *)v6 = v19;
            *(_DWORD *)(v6 + 8) = *(_DWORD *)(v17 + 8);
            *v18 = v6;
            *(_DWORD *)(v25 + v15) = *(_DWORD *)(v24 + v15);
            v15 += 4;
            if ( v27 == i )
              goto LABEL_13;
            goto LABEL_6;
          }
        }
        v29 = v6;
        sub_16BD1C0("Allocation failed");
        v6 = v29;
      }
      v21 = (void *)(v6 + 16);
      if ( v30 <= 1 )
        goto LABEL_5;
      goto LABEL_11;
    }
  }
LABEL_2:
  result = *(unsigned __int8 *)(v2 + 240);
  *(_BYTE *)(v3 + 240) = result;
  return result;
}
