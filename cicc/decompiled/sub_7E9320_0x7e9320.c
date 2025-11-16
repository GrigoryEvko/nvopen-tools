// Function: sub_7E9320
// Address: 0x7e9320
//
void __fastcall sub_7E9320(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  _QWORD *j; // r12
  __int64 i; // rax
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // rax
  bool v20; // dl
  _QWORD *v21; // rcx
  __int64 *v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // rcx
  __int64 v27; // r14
  char v28; // dl
  __int64 v29; // rdi
  char v30; // dl
  __int64 v31; // r12
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  bool v37; // dl
  _QWORD *v38; // rcx
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  char v44; // dl
  __int64 v45; // r12
  __int64 v46; // [rsp+8h] [rbp-58h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  int v50; // [rsp+28h] [rbp-38h]
  __int64 v51; // [rsp+28h] [rbp-38h]

  if ( !a3 )
    goto LABEL_2;
  v25 = *(_QWORD *)(a1 + 104);
  if ( !v25 )
    goto LABEL_2;
  if ( (*(_BYTE *)(a2 + 89) & 4) == 0 || (*(_BYTE *)(a2 + 203) & 1) != 0 )
  {
    v51 = 0;
    v26 = 0;
  }
  else
  {
    v41 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL);
    v51 = v41;
    if ( (*(_BYTE *)(v41 + 89) & 4) != 0 )
    {
      do
        v41 = *(_QWORD *)(*(_QWORD *)(v41 + 40) + 32LL);
      while ( (*(_BYTE *)(v41 + 89) & 4) != 0 );
      v51 = v41;
    }
    v26 = 0;
    v42 = *(_QWORD *)(*(_QWORD *)(v51 + 168) + 216LL);
    if ( v42 )
    {
      do
      {
        v43 = v42;
        v42 = *(_QWORD *)(v42 + 112);
      }
      while ( v42 );
      v26 = v43;
    }
  }
  v46 = 0;
  v27 = v26;
  v47 = 0;
  v49 = 0;
  do
  {
    while ( 1 )
    {
      v30 = *(_BYTE *)(v25 + 140);
      v31 = v25;
      v25 = *(_QWORD *)(v25 + 112);
      if ( (unsigned __int8)(v30 - 9) <= 2u )
      {
        v49 += (*(_BYTE *)(*(_QWORD *)(v31 + 168) + 109LL) & 0x28) == 0;
        break;
      }
      ++v49;
      if ( v30 != 12 || !*(_QWORD *)(v31 + 8) || (*(_BYTE *)(v31 + 186) & 1) == 0 )
        break;
      if ( v47 )
        *(_QWORD *)(v46 + 112) = v31;
      else
        v47 = v31;
      *(_QWORD *)(v31 + 112) = 0;
      v46 = v31;
      if ( !v25 )
        goto LABEL_69;
    }
    sub_813140(v31, 6, 0, a2, a1);
    sub_72B850(v31);
    *(_BYTE *)(v31 + 141) |= 8u;
    if ( v51 )
    {
      if ( v27 )
        *(_QWORD *)(v27 + 112) = v31;
      else
        *(_QWORD *)(*(_QWORD *)(v51 + 168) + 216LL) = v31;
      *(_QWORD *)(v31 + 112) = 0;
      v27 = v31;
    }
    else
    {
      sub_7365B0(v31, 0);
    }
    v28 = *(_BYTE *)(v31 + 140);
    if ( v28 == 2 )
    {
      v44 = *(_BYTE *)(v31 + 161);
      if ( (v44 & 8) != 0 && (**(_BYTE **)(v31 + 176) & 1) != 0 )
      {
        v45 = *(_QWORD *)(v31 + 168);
        if ( (v44 & 0x10) != 0 )
          v45 = *(_QWORD *)(v45 + 96);
        while ( v45 )
        {
          sub_813140(v45, 2, 1, a2, a1);
          *(_BYTE *)(v45 + 89) &= ~1u;
          v45 = *(_QWORD *)(v45 + 120);
        }
      }
    }
    else if ( (unsigned __int8)(v28 - 9) <= 2u )
    {
      v29 = *(_QWORD *)(*(_QWORD *)(v31 + 168) + 152LL);
      if ( v29 )
      {
        if ( (*(_BYTE *)(v29 + 29) & 0x20) == 0 )
          sub_814600();
      }
    }
  }
  while ( v25 );
LABEL_69:
  v32 = v49;
  *(_QWORD *)(a1 + 104) = v47;
  v33 = sub_85EB10(a1);
  if ( v33 )
    *(_QWORD *)(v33 + 32) = 0;
  v34 = *(_QWORD *)(a1 + 80);
  if ( v34 )
  {
    v35 = v34 + 72;
    v36 = *(_QWORD *)(v34 + 72);
    if ( v36 )
    {
      if ( v49 )
      {
        do
        {
          if ( *(_BYTE *)(v36 + 40) == 20 )
          {
            v38 = (_QWORD *)(v36 + 72);
            v39 = *(__int64 **)(v36 + 72);
            if ( v39 )
            {
              if ( v32 )
              {
                while ( 1 )
                {
                  v40 = (__int64 *)*v39;
                  if ( *((_BYTE *)v39 + 8) == 6 )
                  {
                    *v38 = v40;
                    --v32;
                    if ( !v40 )
                      goto LABEL_106;
                    if ( !v32 )
                      break;
                  }
                  else
                  {
                    v38 = v39;
                    if ( !v40 )
                      goto LABEL_106;
                  }
                  v39 = v40;
                }
              }
              v37 = 1;
              v32 = 0;
            }
            else
            {
LABEL_106:
              v37 = v32 == 0;
            }
            v36 = *(_QWORD *)(*(_QWORD *)v35 + 16LL);
            if ( *(_QWORD *)(*(_QWORD *)v35 + 72LL) )
              v35 = *(_QWORD *)v35 + 16LL;
            else
              *(_QWORD *)v35 = v36;
          }
          else
          {
            v35 = v36 + 16;
            v36 = *(_QWORD *)(v36 + 16);
            v37 = v32 == 0;
          }
        }
        while ( v36 && !v37 );
      }
    }
  }
LABEL_2:
  v5 = *(_QWORD *)(a1 + 112);
  if ( v5 )
  {
    v6 = *(int *)(a1 + 240);
    *(_QWORD *)(a1 + 112) = 0;
    v50 = v6;
    if ( (_DWORD)v6 != -1 )
    {
      v7 = qword_4F04C68[0] + 776 * v6;
      v8 = *(_QWORD *)(v7 + 24);
      v9 = v7 + 32;
      if ( !v8 )
        v8 = v9;
      *(_QWORD *)(v8 + 40) = 0;
    }
    v10 = 0;
    while ( 1 )
    {
      if ( *(_DWORD *)(v5 + 64)
        && (*(_QWORD *)(v5 + 168) & 0x200010000LL) == 0
        && ((*(_BYTE *)(v5 + 176) & 0x40) == 0 || *(_BYTE *)(*(_QWORD *)(v5 + 120) + 140LL) != 14) )
      {
        ++v10;
      }
      v11 = *(_QWORD *)(v5 + 112);
      sub_7E7700(v5, a1, a2);
      if ( !v11 )
        break;
      v5 = v11;
    }
    if ( v50 != -1 )
    {
      for ( i = *(_QWORD *)(a1 + 112); i && *(_QWORD *)(i + 112); i = *(_QWORD *)(i + 112) )
        ;
      v14 = qword_4F04C68[0] + 776LL * v50;
      v15 = *(_QWORD *)(v14 + 24);
      v16 = v14 + 32;
      if ( !v15 )
        v15 = v16;
      *(_QWORD *)(v15 + 40) = i;
    }
    v17 = *(_QWORD *)(a1 + 80);
    if ( v17 )
    {
      v18 = (__int64 *)(v17 + 72);
      v19 = *(_QWORD *)(v17 + 72);
      if ( v19 )
      {
        if ( v10 )
        {
          do
          {
            if ( *(_BYTE *)(v19 + 40) != 20 )
            {
              v18 = (__int64 *)(v19 + 16);
              v19 = *(_QWORD *)(v19 + 16);
              v20 = v10 == 0;
              continue;
            }
            v21 = (_QWORD *)(v19 + 72);
            v22 = *(__int64 **)(v19 + 72);
            if ( v22 )
            {
              if ( v10 )
              {
                while ( 1 )
                {
                  v23 = (__int64 *)*v22;
                  if ( *((_BYTE *)v22 + 8) == 7 && *(_BYTE *)(v22[2] + 136) <= 2u )
                  {
                    *v21 = v23;
                    --v10;
                    if ( !v23 )
                      goto LABEL_45;
                    if ( !v10 )
                      break;
                  }
                  else
                  {
                    v21 = v22;
                    if ( !v23 )
                      goto LABEL_45;
                  }
                  v22 = v23;
                }
              }
              v24 = *v18;
              v10 = 0;
              v20 = 1;
              v19 = *(_QWORD *)(*v18 + 16);
              if ( *(_QWORD *)(*v18 + 72) )
              {
LABEL_44:
                v18 = (__int64 *)(v24 + 16);
                continue;
              }
            }
            else
            {
LABEL_45:
              v24 = *v18;
              v20 = v10 == 0;
              v19 = *(_QWORD *)(*v18 + 16);
              if ( *(_QWORD *)(*v18 + 72) )
                goto LABEL_44;
            }
            *v18 = v19;
          }
          while ( v19 && !v20 );
        }
      }
    }
  }
  *(_BYTE *)(a2 + 206) |= 1u;
  for ( j = *(_QWORD **)(a1 + 160); j; j = (_QWORD *)*j )
    sub_7E9320(j, a2, a3);
}
