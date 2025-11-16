// Function: sub_1F04A30
// Address: 0x1f04a30
//
unsigned __int64 __fastcall sub_1F04A30(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  unsigned int v4; // r15d
  __int64 v5; // rbx
  __int64 v6; // r8
  unsigned __int16 *v7; // r9
  unsigned __int64 result; // rax
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // rdx
  unsigned __int64 v11; // rbx
  __int16 *v12; // rcx
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // esi
  _QWORD *v16; // r15
  __int64 v17; // rdx
  unsigned __int16 v18; // r10
  __int64 v19; // rax
  unsigned int v20; // ecx
  __int16 v21; // r11
  unsigned __int16 v22; // si
  _WORD *v23; // rdi
  unsigned __int16 *v24; // rcx
  unsigned __int16 v25; // r11
  unsigned __int16 *v26; // rdi
  unsigned __int16 *v27; // rcx
  unsigned __int16 v28; // r14
  unsigned int v29; // edi
  unsigned int v30; // eax
  _DWORD *v31; // rdx
  __int64 v32; // rax
  _DWORD *v33; // rax
  __int16 v34; // ax
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 i; // rbx
  unsigned __int16 *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // [rsp+0h] [rbp-60h]
  _QWORD *v41; // [rsp+8h] [rbp-58h]
  __int64 *v42; // [rsp+10h] [rbp-50h]
  __int16 *v43; // [rsp+18h] [rbp-48h]
  unsigned __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v42 = (__int64 *)(a1 + 2032);
  v3 = *(_QWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 2048) = 0;
  *(_QWORD *)(a1 + 2032) = v3;
  v4 = *(_DWORD *)(v3 + 16);
  if ( v4 < *(_DWORD *)(a1 + 2096) >> 2 || v4 > *(_DWORD *)(a1 + 2096) )
  {
    _libc_free(*(_QWORD *)(a1 + 2088));
    v5 = (__int64)_libc_calloc(v4, 1u);
    if ( !v5 && (v4 || (v5 = malloc(1u)) == 0) )
      sub_16BD1C0("Allocation failed", 1u);
    *(_QWORD *)(a1 + 2088) = v5;
    *(_DWORD *)(a1 + 2096) = v4;
  }
  sub_1DC2AE0(v42, a2);
  v41 = a2 + 3;
  if ( (a2[3] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    BUG();
  v44 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  result = *(_QWORD *)v44;
  if ( (*(_QWORD *)v44 & 4) == 0 && (*(_BYTE *)((a2[3] & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(result + 46) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
    v44 = result;
    goto LABEL_16;
  }
LABEL_7:
  if ( v41 == (_QWORD *)v44 )
    return result;
  do
  {
    if ( (unsigned __int16)(**(_WORD **)(v44 + 16) - 12) <= 1u )
      goto LABEL_9;
    v40 = v44;
    v11 = v44;
    if ( (*(_BYTE *)(v44 + 46) & 4) != 0 )
    {
      do
        v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v11 + 46) & 4) != 0 );
    }
    else
    {
      v11 = v44;
    }
    v45 = *(_QWORD *)(v11 + 32);
    v12 = (__int16 *)(*(_QWORD *)(v44 + 24) + 24LL);
    v43 = v12;
    v13 = v45 + 40LL * *(unsigned int *)(v11 + 40);
    if ( v45 != v13 )
    {
      while ( 1 )
      {
LABEL_26:
        if ( *(_BYTE *)v45 )
        {
          if ( *(_BYTE *)v45 == 12 )
          {
            sub_1DC1EF0((__int64)v42, v45, 0, (__int64)v12, v6, (int)v7);
            v35 = v13;
            v36 = v45 + 40;
            if ( v45 + 40 == v13 )
              goto LABEL_50;
            goto LABEL_68;
          }
        }
        else if ( (*(_BYTE *)(v45 + 3) & 0x10) != 0 )
        {
          v15 = *(_DWORD *)(v45 + 8);
          if ( v15 )
          {
            v16 = *(_QWORD **)(a1 + 2032);
            if ( !v16 )
              BUG();
            v17 = v16[1];
            v18 = 0;
            v19 = v16[7];
            v20 = *(_DWORD *)(v17 + 24LL * v15 + 16);
            v21 = v15 * (v20 & 0xF);
            v22 = 0;
            v23 = (_WORD *)(v19 + 2LL * (v20 >> 4));
            v24 = v23 + 1;
            v25 = *v23 + v21;
LABEL_31:
            v26 = v24;
            while ( 1 )
            {
              v7 = v26;
              if ( !v26 )
              {
                v28 = v18;
                v12 = 0;
                goto LABEL_35;
              }
              v27 = (unsigned __int16 *)(v16[6] + 4LL * v25);
              v28 = *v27;
              v22 = v27[1];
              if ( *v27 )
                break;
LABEL_79:
              LODWORD(v6) = *v26;
              v24 = 0;
              ++v26;
              if ( !(_WORD)v6 )
                goto LABEL_31;
              v25 += v6;
            }
            while ( 1 )
            {
              v12 = (__int16 *)(v19 + 2LL * *(unsigned int *)(v17 + 24LL * v28 + 8));
              if ( v12 )
                break;
              if ( !v22 )
              {
                v18 = v28;
                goto LABEL_79;
              }
              v28 = v22;
              v22 = 0;
            }
LABEL_35:
            while ( v7 )
            {
              while ( 1 )
              {
                v29 = *(_DWORD *)(a1 + 2048);
                v30 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2088) + v28);
                if ( v30 < v29 )
                {
                  v6 = *(_QWORD *)(a1 + 2040);
                  while ( 1 )
                  {
                    v31 = (_DWORD *)(v6 + 4LL * v30);
                    if ( v28 == *v31 )
                      break;
                    v30 += 256;
                    if ( v29 <= v30 )
                      goto LABEL_44;
                  }
                  v32 = 4LL * v29;
                  if ( v31 != (_DWORD *)(v6 + v32) )
                  {
                    v33 = (_DWORD *)(v6 + v32 - 4);
                    if ( v31 != v33 )
                    {
                      *v31 = *v33;
                      v6 = *(_QWORD *)(a1 + 2040);
                      *(_BYTE *)(*(_QWORD *)(a1 + 2088) + *(unsigned int *)(v6 + 4LL * *(unsigned int *)(a1 + 2048) - 4)) = ((__int64)v31 - v6) >> 2;
                      v29 = *(_DWORD *)(a1 + 2048);
                    }
                    *(_DWORD *)(a1 + 2048) = v29 - 1;
                  }
                }
LABEL_44:
                v34 = *v12++;
                if ( !v34 )
                  break;
                v28 += v34;
              }
              if ( v22 )
              {
                v39 = v22;
                v28 = v22;
                v22 = 0;
                v12 = (__int16 *)(v16[7] + 2LL * *(unsigned int *)(v16[1] + 24 * v39 + 8));
              }
              else
              {
                v22 = *v7;
                v25 += *v7;
                if ( *v7 )
                {
                  ++v7;
                  v38 = (unsigned __int16 *)(v16[6] + 4LL * v25);
                  v28 = *v38;
                  v22 = v38[1];
                  v12 = (__int16 *)(v16[7] + 2LL * *(unsigned int *)(v16[1] + 24LL * *v38 + 8));
                }
                else
                {
                  v12 = 0;
                  v7 = 0;
                }
              }
            }
          }
        }
        v35 = v13;
        v36 = v45 + 40;
        if ( v45 + 40 == v13 )
        {
LABEL_50:
          while ( 1 )
          {
            v11 = *(_QWORD *)(v11 + 8);
            if ( v43 == (__int16 *)v11 || (*(_BYTE *)(v11 + 46) & 4) == 0 )
              break;
            v13 = *(_QWORD *)(v11 + 32);
            v35 = v13 + 40LL * *(unsigned int *)(v11 + 40);
            if ( v13 != v35 )
              goto LABEL_69;
          }
          v45 = v13;
          v13 = v35;
          if ( v45 == v35 )
            goto LABEL_52;
        }
        else
        {
LABEL_68:
          v13 = v36;
LABEL_69:
          v45 = v13;
          v13 = v35;
        }
      }
    }
    v14 = *(_QWORD *)(v11 + 32);
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( v12 == (__int16 *)v11 || (*(_BYTE *)(v11 + 46) & 4) == 0 )
        break;
      v14 = *(_QWORD *)(v11 + 32);
      v13 = v14 + 40LL * *(unsigned int *)(v11 + 40);
      if ( v14 != v13 )
      {
        v45 = *(_QWORD *)(v11 + 32);
        goto LABEL_26;
      }
    }
    v45 = v14;
    if ( v14 != v13 )
      goto LABEL_26;
LABEL_52:
    if ( (*(_BYTE *)(v44 + 46) & 0xC) != 0 )
    {
      if ( **(_WORD **)(v44 + 16) == 16 )
      {
        sub_1F03D80(*(_QWORD *)(a1 + 40), v42, *(_QWORD *)(v44 + 32), *(_DWORD *)(v44 + 40), 0);
        v40 = *(_QWORD *)(v44 + 8);
      }
      for ( i = *(_QWORD *)(v40 + 8); (*(_BYTE *)(i + 46) & 8) != 0; i = *(_QWORD *)(i + 8) )
        ;
      do
      {
        if ( (unsigned __int16)(**(_WORD **)(i + 16) - 12) > 1u )
          sub_1F03D80(*(_QWORD *)(a1 + 40), v42, *(_QWORD *)(i + 32), *(_DWORD *)(i + 40), 1);
        i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL;
      }
      while ( v40 != i );
    }
    else
    {
      sub_1F03D80(*(_QWORD *)(a1 + 40), v42, *(_QWORD *)(v44 + 32), *(_DWORD *)(v44 + 40), 1);
    }
LABEL_9:
    v9 = (unsigned __int64 *)(*(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL);
    v10 = v9;
    if ( !v9 )
      BUG();
    v44 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v9;
    if ( (result & 4) != 0 || (*((_BYTE *)v10 + 46) & 4) == 0 )
      goto LABEL_7;
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(result + 46) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
    v44 = result;
LABEL_16:
    ;
  }
  while ( v41 != (_QWORD *)v44 );
  return result;
}
