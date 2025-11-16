// Function: sub_386AC60
// Address: 0x386ac60
//
__int64 __fastcall sub_386AC60(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // ecx
  int v13; // ecx
  unsigned int v14; // r9d
  __int64 *v15; // rdi
  __int64 v16; // r10
  __int64 v17; // rcx
  unsigned __int8 v18; // di
  __int64 v19; // rax
  int v20; // r9d
  __int64 v21; // r8
  unsigned int v22; // ecx
  __int64 *v23; // rdi
  __int64 v24; // r10
  __int64 *v25; // r10
  unsigned int v26; // eax
  __int64 *v27; // rcx
  __int64 v28; // r11
  __int64 v29; // r9
  int v30; // ecx
  unsigned int v31; // r10d
  __int64 *v32; // rdi
  __int64 v33; // r11
  __int64 v34; // rdi
  __int64 v35; // rbx
  __int64 v36; // rdi
  int v37; // edi
  int v38; // edi
  int v39; // r11d
  int v40; // edi
  int v41; // r11d
  int v42; // ecx
  int v43; // ebx
  int v44; // ebx
  __int64 v45[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_QWORD *)(a2 - 48);
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(_BYTE *)(v6 + 16);
  if ( v8 > 0x10u )
  {
    v29 = *(_QWORD *)(a1 + 40);
    v30 = *(_DWORD *)(v29 + 24);
    if ( !v30 )
    {
      if ( *(_BYTE *)(v7 + 16) <= 0x10u )
        return sub_386A280(a1, (__int64 *)a2, a3, a4);
      goto LABEL_12;
    }
    v13 = v30 - 1;
    v11 = *(_QWORD *)(v29 + 8);
    v31 = v13 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v32 = (__int64 *)(v11 + 16LL * v31);
    v33 = *v32;
    if ( v6 == *v32 )
    {
LABEL_23:
      v34 = v32[1];
      if ( !v34 )
      {
        if ( *(_BYTE *)(v7 + 16) <= 0x10u )
          return sub_386A280(a1, (__int64 *)a2, a3, a4);
        v11 = *(_QWORD *)(v29 + 8);
        v12 = *(_DWORD *)(v29 + 24);
LABEL_6:
        v13 = v12 - 1;
        goto LABEL_7;
      }
      v8 = *(_BYTE *)(v34 + 16);
    }
    else
    {
      v37 = 1;
      while ( v33 != -8 )
      {
        v44 = v37 + 1;
        v31 = v13 & (v37 + v31);
        v32 = (__int64 *)(v11 + 16LL * v31);
        v33 = *v32;
        if ( v6 == *v32 )
          goto LABEL_23;
        v37 = v44;
      }
      v34 = v6;
    }
    if ( *(_BYTE *)(v7 + 16) <= 0x10u )
    {
      if ( v8 <= 0x10u )
      {
        v6 = v34;
        goto LABEL_3;
      }
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
    }
    v6 = v34;
LABEL_7:
    v14 = v13 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v7 == *v15 )
    {
LABEL_8:
      v17 = v15[1];
      if ( v17 )
      {
        v18 = *(_BYTE *)(v17 + 16);
        if ( v8 > 0x10u )
        {
          if ( v18 <= 0x10u )
            return sub_386A280(a1, (__int64 *)a2, a3, a4);
          v7 = v17;
          goto LABEL_12;
        }
LABEL_33:
        v7 = v17;
        goto LABEL_34;
      }
    }
    else
    {
      v38 = 1;
      while ( v16 != -8 )
      {
        v39 = v38 + 1;
        v14 = v13 & (v38 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
          goto LABEL_8;
        v38 = v39;
      }
    }
    if ( v8 <= 0x10u )
    {
LABEL_32:
      v18 = *(_BYTE *)(v7 + 16);
      v17 = *(_QWORD *)(a2 - 24);
      goto LABEL_33;
    }
LABEL_12:
    v19 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v23 = (__int64 *)(v21 + 24LL * v22);
      v24 = *v23;
      if ( v6 == *v23 )
      {
LABEL_14:
        v25 = (__int64 *)(v21 + 24 * v19);
        if ( v23 != v25 )
        {
          v26 = v20 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v27 = (__int64 *)(v21 + 24LL * v26);
          v28 = *v27;
          if ( v7 == *v27 )
          {
LABEL_16:
            if ( v27 != v25 && v23[1] == v27[1] )
            {
              v6 = v23[2];
              v7 = v27[2];
            }
          }
          else
          {
            v42 = 1;
            while ( v28 != -8 )
            {
              v43 = v42 + 1;
              v26 = v20 & (v42 + v26);
              v27 = (__int64 *)(v21 + 24LL * v26);
              v28 = *v27;
              if ( v7 == *v27 )
                goto LABEL_16;
              v42 = v43;
            }
          }
        }
      }
      else
      {
        v40 = 1;
        while ( v24 != -8 )
        {
          v41 = v40 + 1;
          v22 = v20 & (v40 + v22);
          v23 = (__int64 *)(v21 + 24LL * v22);
          v24 = *v23;
          if ( *v23 == v6 )
            goto LABEL_14;
          v40 = v41;
        }
      }
    }
    if ( *(_BYTE *)(v6 + 16) > 0x10u )
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
    v18 = *(_BYTE *)(v7 + 16);
LABEL_34:
    if ( v18 <= 0x10u )
      goto LABEL_3;
    return sub_386A280(a1, (__int64 *)a2, a3, a4);
  }
  if ( *(_BYTE *)(v7 + 16) > 0x10u )
  {
    v10 = *(_QWORD *)(a1 + 40);
    v11 = *(_QWORD *)(v10 + 8);
    v12 = *(_DWORD *)(v10 + 24);
    if ( !v12 )
      goto LABEL_32;
    goto LABEL_6;
  }
LABEL_3:
  if ( *(_QWORD *)v7 != *(_QWORD *)v6 )
    return sub_386A280(a1, (__int64 *)a2, a3, a4);
  v35 = sub_15A37B0(*(_WORD *)(a2 + 18) & 0x7FFF, (_QWORD *)v6, (_QWORD *)v7, 0);
  if ( !v35 )
    return sub_386A280(a1, (__int64 *)a2, a3, a4);
  v36 = *(_QWORD *)(a1 + 40);
  v45[0] = a2;
  sub_38526A0(v36, v45)[1] = v35;
  return 1;
}
