// Function: sub_1AC61E0
// Address: 0x1ac61e0
//
__int64 __fastcall sub_1AC61E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // ecx
  __int64 v5; // rsi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v10; // eax
  char v11; // dl
  __int16 v12; // dx
  __int64 v13; // r8
  unsigned __int8 v14; // r10
  __int64 v15; // r9
  unsigned int v16; // ecx
  unsigned int v17; // edi
  __int64 *v18; // rdx
  __int64 v19; // r11
  __int64 *v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  int v23; // r9d
  __int64 v24; // rdx
  int v25; // edi
  __int64 v26; // r11
  int v27; // edi
  unsigned int v28; // r9d
  __int64 *v29; // rdx
  __int64 v30; // r14
  int v31; // edx
  int v32; // r13d
  int v33; // edx
  int v34; // r13d

  v3 = *(_DWORD *)(a1 + 152);
  v5 = *(_QWORD *)(a1 + 136);
  if ( !v3 )
  {
LABEL_7:
    v7 = (__int64 *)(v5 + 16LL * v3);
    goto LABEL_8;
  }
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v10 = 1;
    while ( v8 != -8 )
    {
      v23 = v10 + 1;
      v6 = (v3 - 1) & (v10 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v10 = v23;
    }
    goto LABEL_7;
  }
LABEL_3:
  if ( v7 != (__int64 *)(v5 + 16LL * v3) )
    return v7[1];
LABEL_8:
  v11 = *(_BYTE *)(a2 + 16);
  if ( v11 == 3 )
  {
    if ( !sub_15E4F60(a2) )
      __asm { jmp     rax }
    return 0;
  }
  if ( v11 != 5 )
    return 0;
  v12 = *(_WORD *)(a2 + 18);
  if ( v12 != 32 )
  {
    if ( v12 == 47 )
    {
      v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v14 = *(_BYTE *)(v13 + 16);
      v15 = v13;
      if ( v14 > 0x10u )
      {
        v24 = *(_QWORD *)(a1 + 48);
        if ( v24 == *(_QWORD *)(a1 + 56) )
          v24 = *(_QWORD *)(*(_QWORD *)(a1 + 72) - 8LL) + 512LL;
        v25 = *(_DWORD *)(v24 - 8);
        v15 = 0;
        if ( v25 )
        {
          v26 = *(_QWORD *)(v24 - 24);
          v27 = v25 - 1;
          v28 = v27 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v13 == *v29 )
          {
LABEL_33:
            v15 = v29[1];
          }
          else
          {
            v33 = 1;
            while ( v30 != -8 )
            {
              v34 = v33 + 1;
              v28 = v27 & (v33 + v28);
              v29 = (__int64 *)(v26 + 16LL * v28);
              v30 = *v29;
              if ( v13 == *v29 )
                goto LABEL_33;
              v33 = v34;
            }
            v15 = 0;
          }
        }
      }
      if ( v3 )
      {
        v16 = v3 - 1;
        v17 = v16 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v18 = (__int64 *)(v5 + 16LL * v17);
        v19 = *v18;
        if ( v15 == *v18 )
        {
LABEL_17:
          if ( v7 != v18 )
          {
            v20 = (__int64 *)v18[1];
            goto LABEL_19;
          }
        }
        else
        {
          v31 = 1;
          while ( v19 != -8 )
          {
            v32 = v31 + 1;
            v17 = v16 & (v31 + v17);
            v18 = (__int64 *)(v5 + 16LL * v17);
            v19 = *v18;
            if ( v15 == *v18 )
              goto LABEL_17;
            v31 = v32;
          }
        }
      }
      if ( v14 == 3 )
      {
        v20 = (__int64 *)sub_1AC5C90(v13);
LABEL_19:
        if ( v20 )
          return sub_14D66F0(v20, **(_QWORD **)(*(_QWORD *)a2 + 16LL), *(_QWORD *)(a1 + 640));
      }
    }
    return 0;
  }
  v21 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v21 + 16) != 3 )
    return 0;
  v22 = sub_1AC5C90(v21);
  if ( !v22 )
    return 0;
  return sub_14D81F0(v22, a2);
}
