// Function: sub_12794A0
// Address: 0x12794a0
//
__int64 __fastcall sub_12794A0(__int64 a1, __int64 a2)
{
  bool v4; // zf
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 *v16; // rax
  unsigned __int8 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rax
  unsigned int i; // eax
  __int64 v21; // rsi
  __int64 v22; // rax
  _QWORD *v23; // rdx
  __int64 *v24; // rsi
  unsigned int v25; // edi
  __int64 *v26; // rcx
  int v27; // r11d
  __int64 *v28; // r14
  int v29; // eax
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // eax
  int v35; // ecx
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rsi
  int v39; // r9d
  __int64 *v40; // r8
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  int v44; // r8d
  unsigned int v45; // r13d
  __int64 *v46; // rdi
  __int64 v47; // rcx
  unsigned int v48; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v49; // [rsp+10h] [rbp-50h] BYREF
  __int64 v50; // [rsp+18h] [rbp-48h]
  __int64 v51; // [rsp+20h] [rbp-40h]

  v4 = *(_BYTE *)(a2 + 140) == 12;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  if ( v4 )
  {
    v5 = a2;
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
    if ( v5 != a2 )
      sub_127B550("error while translating tag type!");
  }
  v6 = *(_DWORD *)(a1 + 48);
  v7 = a1 + 24;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_64;
  }
  v8 = *(_QWORD *)(a1 + 32);
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    v27 = 1;
    v28 = 0;
    while ( v11 != -8 )
    {
      if ( !v28 && v11 == -16 )
        v28 = v10;
      v9 = (v6 - 1) & (v27 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_8;
      ++v27;
    }
    if ( !v28 )
      v28 = v10;
    v29 = *(_DWORD *)(a1 + 40);
    ++*(_QWORD *)(a1 + 24);
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 44) - v30 > v6 >> 3 )
      {
LABEL_54:
        *(_DWORD *)(a1 + 40) = v30;
        if ( *v28 != -8 )
          --*(_DWORD *)(a1 + 44);
        *v28 = a2;
        v28[1] = 0;
        goto LABEL_57;
      }
      sub_1278640(v7, v6);
      v41 = *(_DWORD *)(a1 + 48);
      if ( v41 )
      {
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a1 + 32);
        v44 = 1;
        v45 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v30 = *(_DWORD *)(a1 + 40) + 1;
        v46 = 0;
        v28 = (__int64 *)(v43 + 16LL * v45);
        v47 = *v28;
        if ( *v28 != a2 )
        {
          while ( v47 != -8 )
          {
            if ( !v46 && v47 == -16 )
              v46 = v28;
            v45 = v42 & (v44 + v45);
            v28 = (__int64 *)(v43 + 16LL * v45);
            v47 = *v28;
            if ( *v28 == a2 )
              goto LABEL_54;
            ++v44;
          }
          if ( v46 )
            v28 = v46;
        }
        goto LABEL_54;
      }
LABEL_97:
      ++*(_DWORD *)(a1 + 40);
      BUG();
    }
LABEL_64:
    sub_1278640(v7, 2 * v6);
    v34 = *(_DWORD *)(a1 + 48);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a1 + 32);
      v37 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = *(_DWORD *)(a1 + 40) + 1;
      v28 = (__int64 *)(v36 + 16LL * v37);
      v38 = *v28;
      if ( *v28 != a2 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( v38 == -16 && !v40 )
            v40 = v28;
          v37 = v35 & (v39 + v37);
          v28 = (__int64 *)(v36 + 16LL * v37);
          v38 = *v28;
          if ( *v28 == a2 )
            goto LABEL_54;
          ++v39;
        }
        if ( v40 )
          v28 = v40;
      }
      goto LABEL_54;
    }
    goto LABEL_97;
  }
LABEL_8:
  v12 = v10[1];
  if ( !v12 )
  {
    v28 = v10;
LABEL_57:
    v31 = sub_16440F0(**(_QWORD **)a1);
    v28[1] = v31;
    v12 = v31;
  }
  if ( (*(_BYTE *)(v12 + 9) & 1) != 0 || sub_8D23B0(a2) )
    goto LABEL_10;
  if ( (unsigned __int8)sub_1277C40(a1, a2, v14, v15) )
  {
    v16 = *(__int64 **)(a1 + 168);
    if ( *(__int64 **)(a1 + 176) == v16 )
    {
      v24 = &v16[*(unsigned int *)(a1 + 188)];
      v25 = *(_DWORD *)(a1 + 188);
      if ( v16 != v24 )
      {
        v26 = 0;
        while ( *v16 != a2 )
        {
          if ( *v16 == -2 )
            v26 = v16;
          if ( v24 == ++v16 )
          {
            if ( !v26 )
              goto LABEL_78;
            *v26 = a2;
            --*(_DWORD *)(a1 + 192);
            ++*(_QWORD *)(a1 + 160);
            break;
          }
        }
LABEL_17:
        v17 = sub_127B460(a2);
        if ( HIDWORD(qword_4F077B4)
          && qword_4F077A8 <= 0x9DCFu
          && v17
          && (unsigned __int8)sub_1277550(*(_QWORD *)(a2 + 160)) )
        {
          sub_127B550("Bitfields and field types containing bitfields are not supported in packed structures and unions f"
                      "or device compilation, when using this host compiler!");
        }
        if ( *(_BYTE *)(a2 + 140) == 11 )
        {
          sub_127A060(a1, a2, &v49);
        }
        else
        {
          v48 = 0;
          sub_1278D70((_QWORD *)a1, a2, (__int64)&v49, &v48);
        }
        v18 = *(_QWORD **)(a1 + 168);
        if ( *(_QWORD **)(a1 + 176) == v18 )
        {
          v23 = &v18[*(unsigned int *)(a1 + 188)];
          if ( v18 == v23 )
          {
LABEL_77:
            v18 = v23;
          }
          else
          {
            while ( *v18 != a2 )
            {
              if ( v23 == ++v18 )
                goto LABEL_77;
            }
          }
        }
        else
        {
          v18 = (_QWORD *)sub_16CC9F0(a1 + 160, a2);
          if ( *v18 == a2 )
          {
            v32 = *(_QWORD *)(a1 + 176);
            if ( v32 == *(_QWORD *)(a1 + 168) )
              v33 = *(unsigned int *)(a1 + 188);
            else
              v33 = *(unsigned int *)(a1 + 184);
            v23 = (_QWORD *)(v32 + 8 * v33);
          }
          else
          {
            v19 = *(_QWORD *)(a1 + 176);
            if ( v19 != *(_QWORD *)(a1 + 168) )
            {
LABEL_27:
              sub_1643FB0(v12, v49, (v50 - v49) >> 3, v17);
              if ( *(_DWORD *)(a1 + 188) == *(_DWORD *)(a1 + 192) )
              {
                for ( i = *(_DWORD *)(a1 + 240); i; i = *(_DWORD *)(a1 + 240) )
                {
                  v21 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8LL * i - 8);
                  *(_DWORD *)(a1 + 240) = i - 1;
                  sub_12794A0(a1, v21);
                }
              }
              goto LABEL_10;
            }
            v18 = (_QWORD *)(v19 + 8LL * *(unsigned int *)(a1 + 188));
            v23 = v18;
          }
        }
        if ( v23 != v18 )
        {
          *v18 = -2;
          ++*(_DWORD *)(a1 + 192);
        }
        goto LABEL_27;
      }
LABEL_78:
      if ( v25 < *(_DWORD *)(a1 + 184) )
      {
        *(_DWORD *)(a1 + 188) = v25 + 1;
        *v24 = a2;
        ++*(_QWORD *)(a1 + 160);
        goto LABEL_17;
      }
    }
    sub_16CCBA0(a1 + 160, a2);
    goto LABEL_17;
  }
  v22 = *(unsigned int *)(a1 + 240);
  if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 244) )
  {
    sub_16CD150(a1 + 232, a1 + 248, 0, 8);
    v22 = *(unsigned int *)(a1 + 240);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8 * v22) = a2;
  ++*(_DWORD *)(a1 + 240);
LABEL_10:
  if ( v49 )
    j_j___libc_free_0(v49, v51 - v49);
  return v12;
}
