// Function: sub_25A0950
// Address: 0x25a0950
//
__int64 __fastcall sub_25A0950(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  _BYTE *v5; // r12
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 (__fastcall *v10)(__int64); // rax
  char v11; // al
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // r15
  __int64 v16; // r8
  unsigned int v17; // edx
  __int64 *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rax
  _DWORD *v22; // r14
  unsigned int v23; // esi
  __int64 v24; // r9
  __int64 *v25; // r11
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v30; // rax
  int v31; // edi
  int v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+28h] [rbp-48h] BYREF
  unsigned __int64 v35[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = *a1;
  sub_250D230(v35, a2, 5, 0);
  v4 = sub_251C7D0(v3, v35[0], v35[1], a1[1], 0, 0, 1);
  if ( v4 )
  {
    v5 = (_BYTE *)v4;
    v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 128LL);
    if ( v6 == sub_2534E50 )
      v7 = v5[169];
    else
      v7 = v6((__int64)v5);
    if ( v7 )
    {
      v8 = a1[1];
      v9 = (_DWORD *)a1[2];
      if ( !*(_BYTE *)(v8 + 168) )
        *v9 = 0;
      if ( !*(_BYTE *)(v8 + 169) )
        *v9 = 0;
      *(_WORD *)(v8 + 168) = 257;
    }
    v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 120LL);
    if ( v10 == sub_2534E40 )
      v11 = v5[168];
    else
      v11 = v10((__int64)v5);
    if ( v11 )
    {
      v30 = a1[1];
      if ( !*(_BYTE *)(v30 + 168) )
        *(_DWORD *)a1[2] = 0;
      *(_BYTE *)(v30 + 168) = 1;
    }
    v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 112LL);
    if ( v12 == sub_2534E30 )
      v13 = (__int64)(v5 + 120);
    else
      v13 = v12((__int64)v5);
    v14 = *(__int64 **)(v13 + 32);
    v15 = &v14[*(unsigned int *)(v13 + 40)];
    if ( v14 == v15 )
      return 1;
    while ( 1 )
    {
      v20 = a1[1];
      v21 = *v14;
      v22 = (_DWORD *)a1[2];
      v23 = *(_DWORD *)(v20 + 144);
      v34 = *v14;
      v24 = v20 + 120;
      if ( !v23 )
        break;
      v16 = *(_QWORD *)(v20 + 128);
      v17 = (v23 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v18 = (__int64 *)(v16 + 8LL * v17);
      v19 = *v18;
      if ( v21 == *v18 )
      {
LABEL_18:
        if ( v15 == ++v14 )
          return 1;
      }
      else
      {
        v32 = 1;
        v25 = 0;
        while ( v19 != -4096 )
        {
          if ( v25 || v19 != -8192 )
            v18 = v25;
          v17 = (v23 - 1) & (v32 + v17);
          v19 = *(_QWORD *)(v16 + 8LL * v17);
          if ( v21 == v19 )
            goto LABEL_18;
          ++v32;
          v25 = v18;
          v18 = (__int64 *)(v16 + 8LL * v17);
        }
        if ( !v25 )
          v25 = v18;
        v35[0] = (unsigned __int64)v25;
        v31 = *(_DWORD *)(v20 + 136);
        ++*(_QWORD *)(v20 + 120);
        v26 = v31 + 1;
        if ( 4 * (v31 + 1) < 3 * v23 )
        {
          if ( v23 - *(_DWORD *)(v20 + 140) - v26 > v23 >> 3 )
            goto LABEL_23;
          goto LABEL_22;
        }
LABEL_21:
        v23 *= 2;
LABEL_22:
        sub_A35F10(v20 + 120, v23);
        sub_A2AFD0(v20 + 120, &v34, v35);
        v21 = v34;
        v25 = (__int64 *)v35[0];
        v26 = *(_DWORD *)(v20 + 136) + 1;
LABEL_23:
        *(_DWORD *)(v20 + 136) = v26;
        if ( *v25 != -4096 )
          --*(_DWORD *)(v20 + 140);
        *v25 = v21;
        v27 = *(unsigned int *)(v20 + 160);
        v28 = v34;
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 164) )
        {
          v33 = v34;
          sub_C8D5F0(v20 + 152, (const void *)(v20 + 168), v27 + 1, 8u, v34, v24);
          v27 = *(unsigned int *)(v20 + 160);
          v28 = v33;
        }
        ++v14;
        *(_QWORD *)(*(_QWORD *)(v20 + 152) + 8 * v27) = v28;
        ++*(_DWORD *)(v20 + 160);
        *v22 = 0;
        if ( v15 == v14 )
          return 1;
      }
    }
    v35[0] = 0;
    ++*(_QWORD *)(v20 + 120);
    goto LABEL_21;
  }
  return 0;
}
