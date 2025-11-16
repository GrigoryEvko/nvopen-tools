// Function: sub_250CDD0
// Address: 0x250cdd0
//
__int64 __fastcall sub_250CDD0(__int64 a1, __int64 *a2, char *a3)
{
  __int64 result; // rax
  unsigned __int8 *v7; // r14
  unsigned __int8 v8; // cl
  unsigned __int64 v9; // rdx
  unsigned __int8 *v10; // r15
  unsigned __int8 v11; // cl
  __int64 v12; // rdx
  unsigned __int8 **v13; // rax
  unsigned __int8 **v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // eax
  int v18; // edx
  unsigned int v19; // eax
  unsigned __int8 *v20; // rsi
  int v21; // edi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  int v25; // edx
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // rsi
  int v29; // r8d
  char v30; // al

  result = 0;
  if ( *(_DWORD *)(a1 + 3556) <= unk_4FEEF68 )
  {
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
      goto LABEL_29;
    v7 = sub_250CBE0(a2, (__int64)a2);
    v8 = sub_2509800(a2);
    if ( v8 <= 7u && ((1LL << v8) & 0xA8) != 0 )
    {
      v9 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*a2 & 3) == 3 )
        v9 = *(_QWORD *)(v9 + 24);
      if ( **(_BYTE **)(v9 - 32) == 25 )
        goto LABEL_29;
    }
    v10 = sub_250CBE0(a2, (__int64)a2);
    v11 = sub_2509800(a2);
    if ( v11 > 6u || ((1LL << v11) & 0x54) == 0 || !sub_B2FC80((__int64)v10) && !(unsigned __int8)sub_B2FC00(v10) )
      goto LABEL_15;
    v12 = *(_QWORD *)(a1 + 208);
    if ( *(_BYTE *)(v12 + 276) )
    {
      v13 = *(unsigned __int8 ***)(v12 + 256);
      v14 = &v13[*(unsigned int *)(v12 + 268)];
      if ( v13 == v14 )
      {
LABEL_13:
        if ( !*(_QWORD *)(a1 + 4432)
          || !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 *))(a1 + 4440))(a1 + 4416, v10) )
        {
          goto LABEL_29;
        }
        goto LABEL_15;
      }
      while ( v10 != *v13 )
      {
        if ( v14 == ++v13 )
          goto LABEL_13;
      }
    }
    else if ( !sub_C8CA60(v12 + 248, (__int64)v10) )
    {
      goto LABEL_13;
    }
LABEL_15:
    if ( !v7 )
      goto LABEL_34;
    if ( *(_BYTE *)(a1 + 4296) )
      goto LABEL_34;
    v15 = *(_QWORD *)(a1 + 200);
    if ( !*(_DWORD *)(v15 + 40) )
      goto LABEL_34;
    v16 = *(_QWORD *)(v15 + 8);
    v17 = *(_DWORD *)(v15 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = (v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v20 = *(unsigned __int8 **)(v16 + 8LL * v19);
      if ( v7 == v20 )
        goto LABEL_34;
      v21 = 1;
      while ( v20 != (unsigned __int8 *)-4096LL )
      {
        v19 = v18 & (v21 + v19);
        v20 = *(unsigned __int8 **)(v16 + 8LL * v19);
        if ( v7 == v20 )
          goto LABEL_34;
        ++v21;
      }
    }
    v22 = sub_25096F0(a2);
    v23 = *(_QWORD *)(a1 + 200);
    if ( !*(_DWORD *)(v23 + 40) )
      goto LABEL_34;
    v24 = *(_QWORD *)(v23 + 8);
    v25 = *(_DWORD *)(v23 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = (v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v28 = *(_QWORD *)(v24 + 8LL * v27);
      if ( v22 != v28 )
      {
        v29 = 1;
        while ( v28 != -4096 )
        {
          v27 = v26 & (v29 + v27);
          v28 = *(_QWORD *)(v24 + 8LL * v27);
          if ( v22 == v28 )
            goto LABEL_34;
          ++v29;
        }
        goto LABEL_29;
      }
LABEL_34:
      v30 = 1;
      goto LABEL_30;
    }
LABEL_29:
    v30 = 0;
LABEL_30:
    *a3 = v30;
    return 1;
  }
  return result;
}
