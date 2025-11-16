// Function: sub_F581B0
// Address: 0xf581b0
//
void __fastcall sub_F581B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r8
  _QWORD *v12; // rbx
  _QWORD *v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rdx
  unsigned __int8 *v20; // r10
  int v21; // edx
  int v22; // r8d
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rdx
  unsigned __int8 *v28; // r10
  _QWORD *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rdi
  unsigned int v35; // esi
  __int64 v36; // rax
  __int64 v37; // r10
  __int64 v38; // rdx
  int i; // edx
  int v40; // r9d
  unsigned int v41; // edx
  __int64 v42; // r13
  __int64 v43; // rbx
  __int64 v44; // r15
  _QWORD *v45; // rbx
  _QWORD *v46; // rdx
  int v47; // r15d
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // r9
  int v54; // edx
  int v55; // r10d
  __int64 v56; // r9
  int v57; // eax
  int v58; // r9d
  __int64 v59; // [rsp+8h] [rbp-48h]
  __int64 v60; // [rsp+10h] [rbp-40h] BYREF
  __int64 v61; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)a2 != 85 )
    goto LABEL_2;
  v30 = *(_QWORD *)(a2 - 32);
  if ( !v30 )
    goto LABEL_2;
  if ( !*(_BYTE *)v30 && *(_QWORD *)(v30 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v30 + 33) & 0x20) != 0 )
  {
    v41 = *(_DWORD *)(v30 + 36);
    if ( v41 > 0x45 )
    {
      if ( v41 == 71 )
      {
LABEL_54:
        sub_B58E30(&v60, a2);
        v42 = v61;
        v43 = v60;
        if ( v61 == v60 )
        {
LABEL_65:
          if ( *(_BYTE *)a2 != 85 )
            goto LABEL_2;
          v30 = *(_QWORD *)(a2 - 32);
          if ( !v30 )
            goto LABEL_2;
          goto LABEL_39;
        }
        while ( 1 )
        {
          v44 = v43;
          v45 = (_QWORD *)(v43 & 0xFFFFFFFFFFFFFFF8LL);
          v46 = v45;
          v47 = (v44 >> 2) & 1;
          if ( v47 )
            v46 = (_QWORD *)*v45;
          v48 = *(unsigned int *)(a1 + 24);
          if ( !(_DWORD)v48 )
            goto LABEL_63;
          v49 = v46[17];
          v50 = *(_QWORD *)(a1 + 8);
          v51 = (v48 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
          v52 = v50 + ((unsigned __int64)v51 << 6);
          v53 = *(_QWORD *)(v52 + 24);
          if ( v49 == v53 )
            goto LABEL_61;
          v54 = 1;
          if ( v53 != -4096 )
            break;
LABEL_63:
          if ( v47 )
          {
            v43 = (unsigned __int64)(v45 + 1) | 4;
            if ( v42 == v43 )
              goto LABEL_65;
          }
          else
          {
            v43 = (__int64)(v45 + 18);
            if ( v42 == v43 )
              goto LABEL_65;
          }
        }
        while ( 1 )
        {
          v55 = v54 + 1;
          v51 = (v48 - 1) & (v54 + v51);
          v52 = v50 + ((unsigned __int64)v51 << 6);
          v56 = *(_QWORD *)(v52 + 24);
          if ( v49 == v56 )
            break;
          v54 = v55;
          if ( v56 == -4096 )
            goto LABEL_63;
        }
LABEL_61:
        if ( v52 != v50 + (v48 << 6) )
          sub_B59720(a2, v49, *(unsigned __int8 **)(v52 + 56));
        goto LABEL_63;
      }
    }
    else if ( v41 > 0x43 )
    {
      goto LABEL_54;
    }
  }
LABEL_39:
  if ( !*(_BYTE *)v30
    && *(_QWORD *)(v30 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v30 + 33) & 0x20) != 0
    && *(_DWORD *)(v30 + 36) == 68 )
  {
    v31 = sub_B595C0(a2);
    v32 = *(unsigned int *)(a1 + 24);
    v33 = v31;
    if ( (_DWORD)v32 )
    {
      v34 = *(_QWORD *)(a1 + 8);
      v35 = (v32 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v36 = v34 + ((unsigned __int64)v35 << 6);
      v37 = *(_QWORD *)(v36 + 24);
      if ( v33 == v37 )
      {
LABEL_45:
        v38 = v34 + (v32 << 6);
        if ( v36 != v38 )
          sub_B59690(a2, *(_QWORD *)(v36 + 56), v38, v33);
      }
      else
      {
        v57 = 1;
        while ( v37 != -4096 )
        {
          v58 = v57 + 1;
          v35 = (v32 - 1) & (v57 + v35);
          v36 = v34 + ((unsigned __int64)v35 << 6);
          v37 = *(_QWORD *)(v36 + 24);
          if ( v33 == v37 )
            goto LABEL_45;
          v57 = v58;
        }
      }
    }
  }
LABEL_2:
  v4 = *(_QWORD *)(a2 + 64);
  if ( v4 )
  {
    v5 = sub_B14240(v4);
    v7 = v6;
    v8 = v5;
    if ( v5 != v6 )
    {
      while ( *(_BYTE *)(v8 + 32) )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v8 == v6 )
          return;
      }
      if ( v6 != v8 )
      {
LABEL_8:
        sub_B129C0(&v60, v8);
        v9 = v60;
        v10 = v61;
        if ( v61 != v60 )
        {
          v59 = v7;
          do
          {
            while ( 1 )
            {
              v11 = v9;
              v12 = (_QWORD *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
              v13 = v12;
              LODWORD(v11) = (v11 >> 2) & 1;
              v14 = (unsigned int)v11;
              if ( (_DWORD)v11 )
                v13 = (_QWORD *)*v12;
              v15 = *(unsigned int *)(a1 + 24);
              if ( (_DWORD)v15 )
              {
                v16 = (unsigned __int8 *)v13[17];
                v17 = *(_QWORD *)(a1 + 8);
                v18 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
                v19 = v17 + ((unsigned __int64)v18 << 6);
                v20 = *(unsigned __int8 **)(v19 + 24);
                if ( v20 == v16 )
                {
LABEL_16:
                  if ( v19 != v17 + (v15 << 6) )
                    sub_B13360(v8, v16, *(unsigned __int8 **)(v19 + 56), 1);
                }
                else
                {
                  v21 = 1;
                  while ( v20 != (unsigned __int8 *)-4096LL )
                  {
                    v22 = v21 + 1;
                    v18 = (v15 - 1) & (v21 + v18);
                    v19 = v17 + ((unsigned __int64)v18 << 6);
                    v20 = *(unsigned __int8 **)(v19 + 24);
                    if ( v16 == v20 )
                      goto LABEL_16;
                    v21 = v22;
                  }
                }
              }
              if ( v14 || !v12 )
                break;
              v9 = (__int64)(v12 + 18);
              if ( v10 == v9 )
                goto LABEL_20;
            }
            v9 = (unsigned __int64)(v12 + 1) | 4;
          }
          while ( v10 != v9 );
LABEL_20:
          v7 = v59;
        }
        if ( *(_BYTE *)(v8 + 64) == 2 )
        {
          v23 = sub_B13320(v8);
          v24 = *(unsigned int *)(a1 + 24);
          if ( (_DWORD)v24 )
          {
            v25 = *(_QWORD *)(a1 + 8);
            v26 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v27 = v25 + ((unsigned __int64)v26 << 6);
            v28 = *(unsigned __int8 **)(v27 + 24);
            if ( v28 != v23 )
            {
              for ( i = 1; ; i = v40 )
              {
                if ( v28 == (unsigned __int8 *)-4096LL )
                  goto LABEL_24;
                v40 = i + 1;
                v26 = (v24 - 1) & (i + v26);
                v27 = v25 + ((unsigned __int64)v26 << 6);
                v28 = *(unsigned __int8 **)(v27 + 24);
                if ( v23 == v28 )
                  break;
              }
            }
            if ( v27 != v25 + (v24 << 6) )
            {
              v29 = sub_B98A20(*(_QWORD *)(v27 + 56), (__int64)v23);
              sub_B91340(v8 + 40, 1);
              *(_QWORD *)(v8 + 48) = v29;
              sub_B96F50(v8 + 40, 1);
            }
          }
        }
LABEL_24:
        while ( 1 )
        {
          v8 = *(_QWORD *)(v8 + 8);
          if ( v7 == v8 )
            break;
          if ( !*(_BYTE *)(v8 + 32) )
          {
            if ( v8 != v7 )
              goto LABEL_8;
            return;
          }
        }
      }
    }
  }
}
