// Function: sub_1EB7500
// Address: 0x1eb7500
//
__int64 __fastcall sub_1EB7500(__int64 a1, unsigned __int16 a2)
{
  _QWORD *v2; // r11
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v7; // edx
  __int16 v8; // ax
  _WORD *v9; // rcx
  __int16 *v10; // rdx
  unsigned __int16 v11; // r9
  __int16 *v12; // r8
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rsi
  _DWORD *v16; // rdx
  __int16 v17; // ax
  __int64 v18; // rdi
  unsigned int v19; // r13d
  unsigned int v20; // r13d
  __int64 v21; // rsi
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned int v28; // edx
  _WORD *v29; // rdx
  _WORD *v30; // r8
  unsigned __int16 v31; // cx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int16 v35; // cx
  __int16 v36; // ax
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r8
  unsigned int v41; // edx
  __int64 v42; // rsi
  int v43; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v44; // [rsp-60h] [rbp-60h]
  char v45; // [rsp-58h] [rbp-58h]
  unsigned __int16 v46; // [rsp-50h] [rbp-50h]
  _WORD *v47; // [rsp-48h] [rbp-48h]
  int v48; // [rsp-40h] [rbp-40h]
  unsigned __int16 v49; // [rsp-38h] [rbp-38h]
  __int64 v50; // [rsp-30h] [rbp-30h]

  v2 = *(_QWORD **)(a1 + 248);
  if ( !v2 )
    BUG();
  v4 = a2;
  v5 = 24LL * a2;
  v7 = *(_DWORD *)(v2[1] + v5 + 16);
  v8 = v7 & 0xF;
  v9 = (_WORD *)(v2[7] + 2LL * (v7 >> 4));
  v10 = v9 + 1;
  v11 = *v9 + a2 * v8;
LABEL_3:
  v12 = v10;
  while ( v12 )
  {
    v13 = *(unsigned int *)(a1 + 1032);
    v14 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1072) + v11);
    if ( v14 < (unsigned int)v13 )
    {
      v15 = *(_QWORD *)(a1 + 1024);
      while ( 1 )
      {
        v16 = (_DWORD *)(v15 + 4LL * v14);
        if ( v11 == *v16 )
          break;
        v14 += 256;
        if ( (unsigned int)v13 <= v14 )
          goto LABEL_9;
      }
      if ( v16 != (_DWORD *)(v15 + 4 * v13) )
        return (unsigned int)-1;
    }
LABEL_9:
    v17 = *v12;
    v10 = 0;
    ++v12;
    v11 += v17;
    if ( !v17 )
      goto LABEL_3;
  }
  v18 = *(_QWORD *)(a1 + 648);
  v19 = *(_DWORD *)(v18 + 4 * v4);
  switch ( v19 )
  {
    case 1u:
      return 0;
    case 2u:
      return (unsigned int)-1;
    case 0u:
      v45 = 0;
      v46 = 0;
      v47 = 0;
      v49 = 0;
      v50 = 0;
      v26 = v2[7];
      v48 = 0;
      v44 = v2 + 1;
      v27 = v2[1];
      v43 = a2;
      v28 = *(_DWORD *)(v27 + v5 + 16);
      LOWORD(v27) = a2 * (v28 & 0xF);
      v29 = (_WORD *)(v26 + 2LL * (v28 >> 4));
      v30 = v29 + 1;
      v46 = *v29 + v27;
      v47 = v29 + 1;
      while ( v30 )
      {
        v48 = *(_DWORD *)(v2[6] + 4LL * v46);
        v31 = v48;
        if ( (_WORD)v48 )
        {
LABEL_26:
          v32 = *(unsigned int *)(v2[1] + 24LL * v31 + 8);
          v33 = v2[7];
          v49 = v31;
          v34 = v33 + 2 * v32;
          v50 = v34;
          while ( 1 )
          {
            if ( !v34 )
              goto LABEL_31;
            if ( a2 != v49 )
              break;
            v34 += 2;
            v50 = v34;
            v35 = *(_WORD *)(v34 - 2);
            v49 += v35;
            if ( !v35 )
            {
              v50 = 0;
LABEL_31:
              v31 = HIWORD(v48);
              v48 = HIWORD(v48);
              if ( v31 )
                goto LABEL_26;
              goto LABEL_32;
            }
          }
          v37 = *(_DWORD *)(v18 + 4LL * v49);
          if ( v37 == 1 )
            goto LABEL_44;
LABEL_35:
          if ( v37 == 2 )
            return (unsigned int)-1;
          if ( !v37 )
            goto LABEL_42;
          v38 = v37 & 0x7FFFFFFF;
          v39 = *(unsigned int *)(a1 + 400);
          v40 = *(_QWORD *)(a1 + 392);
          v41 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 600) + v38);
          if ( v41 < (unsigned int)v39 )
          {
            while ( 1 )
            {
              v42 = v40 + 24LL * v41;
              if ( (_DWORD)v38 == (*(_DWORD *)(v42 + 8) & 0x7FFFFFFF) )
                break;
              v41 += 256;
              if ( (unsigned int)v39 <= v41 )
                goto LABEL_45;
            }
          }
          else
          {
LABEL_45:
            v42 = v40 + 24 * v39;
          }
          v19 += *(_BYTE *)(v42 + 16) == 0 ? 1 : 100;
LABEL_42:
          while ( 1 )
          {
            sub_1E1D5E0((__int64)&v43);
            if ( !v47 )
              return v19;
            v37 = *(_DWORD *)(*(_QWORD *)(a1 + 648) + 4LL * v49);
            if ( v37 != 1 )
              goto LABEL_35;
LABEL_44:
            ++v19;
          }
        }
LABEL_32:
        v47 = ++v30;
        v36 = *(v30 - 1);
        v46 += v36;
        if ( !v36 )
          return v19;
      }
      return v19;
  }
  v20 = v19 & 0x7FFFFFFF;
  v21 = *(unsigned int *)(a1 + 400);
  v22 = *(_QWORD *)(a1 + 392);
  v23 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 600) + v20);
  if ( v23 < (unsigned int)v21 )
  {
    while ( 1 )
    {
      v24 = v22 + 24LL * v23;
      if ( v20 == (*(_DWORD *)(v24 + 8) & 0x7FFFFFFF) )
        break;
      v23 += 256;
      if ( (unsigned int)v21 <= v23 )
        goto LABEL_15;
    }
  }
  else
  {
LABEL_15:
    v24 = v22 + 24 * v21;
  }
  return *(_BYTE *)(v24 + 16) == 0 ? 1 : 100;
}
