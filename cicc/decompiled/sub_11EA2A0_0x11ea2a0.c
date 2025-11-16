// Function: sub_11EA2A0
// Address: 0x11ea2a0
//
unsigned __int64 __fastcall sub_11EA2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r15
  const char *v8; // r8
  __int64 v9; // r9
  size_t v10; // rdx
  size_t v11; // r14
  unsigned __int64 v12; // r14
  char *v14; // rdi
  unsigned __int64 v15; // r12
  unsigned __int8 *v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 *v22; // rdx
  __int64 v23; // rax
  char *v24; // rax
  __int64 v25; // rdx
  int v26; // eax
  const void *v27; // rsi
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned int v32; // edi
  int *v33; // rax
  int v34; // esi
  int v35; // eax
  int v36; // r11d
  __int64 v37; // [rsp+0h] [rbp-B0h]
  char *v38; // [rsp+8h] [rbp-A8h]
  char v39; // [rsp+8h] [rbp-A8h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  __int16 v41; // [rsp+10h] [rbp-A0h]
  const char *v42; // [rsp+10h] [rbp-A0h]
  __int64 *v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  int v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  _QWORD v47[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v48[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v49[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v50; // [rsp+70h] [rbp-40h]

  v6 = sub_B43CA0(a2);
  v7 = *(_QWORD *)(a2 - 32);
  v43 = (__int64 *)v6;
  if ( v7 )
  {
    if ( *(_BYTE *)v7 )
    {
      v7 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v7 + 24) )
    {
      v7 = 0;
    }
  }
  v8 = sub_BD5D20(v7);
  v11 = v10;
  if ( !*(_BYTE *)(a1 + 80) )
    goto LABEL_6;
  v24 = *(char **)(a1 + 24);
  if ( v24[36] >= 0 )
  {
    v25 = *(_QWORD *)v24;
    v26 = (int)*(unsigned __int8 *)(*(_QWORD *)v24 + 57LL) >> 6;
    if ( v26 )
    {
      if ( v26 == 3 )
      {
        v27 = "amd_vrd2_exp2" + 9;
        v28 = qword_4977328[462];
LABEL_26:
        if ( v28 != v11 )
          goto LABEL_6;
        if ( v11 )
        {
          v42 = v8;
          v29 = memcmp(v8, v27, v11);
          v8 = v42;
          if ( v29 )
            goto LABEL_6;
        }
        goto LABEL_32;
      }
      v30 = *(_QWORD *)(v25 + 144);
      v31 = *(unsigned int *)(v25 + 160);
      if ( (_DWORD)v31 )
      {
        v32 = ((_WORD)v31 - 1) & 0x2163;
        v33 = (int *)(v30 + 40LL * (((_WORD)v31 - 1) & 0x2163));
        v34 = *v33;
        if ( *v33 == 231 )
        {
LABEL_36:
          v27 = (const void *)*((_QWORD *)v33 + 1);
          v28 = *((_QWORD *)v33 + 2);
          goto LABEL_26;
        }
        v35 = 1;
        while ( v34 != -1 )
        {
          v36 = v35 + 1;
          v32 = (v31 - 1) & (v35 + v32);
          v33 = (int *)(v30 + 40LL * v32);
          v34 = *v33;
          if ( *v33 == 231 )
            goto LABEL_36;
          v35 = v36;
        }
      }
      v33 = (int *)(v30 + 40 * v31);
      goto LABEL_36;
    }
  }
  if ( v11 )
  {
LABEL_6:
    v12 = 0;
    goto LABEL_7;
  }
LABEL_32:
  if ( !(unsigned __int8)sub_11E9B60(a1, v43, (__int64)v8, v11, (__int64)v8, v9) )
    goto LABEL_6;
  v12 = sub_11DB650(a2, a3, 0, *(__int64 **)(a1 + 24), 1);
LABEL_7:
  if ( (*(_BYTE *)(v7 + 33) & 0x20) != 0 )
  {
    v14 = *(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( (unsigned __int8)(*v14 - 72) <= 1u )
    {
      v44 = *(_QWORD *)(a2 + 8);
      v15 = sub_11DBA30(v14, a3, *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
      if ( v15 )
      {
        v16 = sub_AD8DD0(v44, 1.0);
        v50 = 257;
        v48[1] = v15;
        LODWORD(v46) = sub_B45210(a2);
        BYTE4(v46) = 1;
        v47[1] = *(_QWORD *)(v15 + 8);
        v48[0] = v16;
        v47[0] = v44;
        v17 = sub_B33D10(a3, 0xD1u, (__int64)v47, 2, (int)v48, 2, v46, (__int64)v49);
        v12 = v17;
        if ( v17 )
        {
          if ( *(_BYTE *)v17 == 85 )
            *(_WORD *)(v17 + 2) = *(_WORD *)(v17 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
        }
      }
    }
  }
  else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1
         && (unsigned __int8)(**(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) - 72) <= 1u )
  {
    v38 = *(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v40 = *(_QWORD *)(a2 + 8);
    if ( sub_11C9D70(v43, *(__int64 **)(a1 + 24), v40, 0x149u, 0x14Au, 0x14Bu) )
    {
      v18 = sub_11DBA30(v38, a3, *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
      if ( v18 )
      {
        v37 = v18;
        v19 = sub_AD8DD0(v40, 1.0);
        v20 = *(_QWORD *)(a3 + 96);
        v21 = (__int64)v19;
        v39 = *(_BYTE *)(a3 + 110);
        v41 = *(_WORD *)(a3 + 108);
        v45 = *(_DWORD *)(a3 + 104);
        *(_DWORD *)(a3 + 104) = sub_B45210(a2);
        v22 = *(__int64 **)(a1 + 24);
        v49[0] = 0;
        v23 = sub_11CD140(v21, v37, v22, 0x149u, 0x14Au, 0x14Bu, a3, v49);
        v12 = v23;
        if ( v23 && *(_BYTE *)v23 == 85 )
          *(_WORD *)(v23 + 2) = *(_WORD *)(v23 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
        *(_QWORD *)(a3 + 96) = v20;
        *(_WORD *)(a3 + 108) = v41;
        *(_DWORD *)(a3 + 104) = v45;
        *(_BYTE *)(a3 + 110) = v39;
      }
    }
  }
  return v12;
}
