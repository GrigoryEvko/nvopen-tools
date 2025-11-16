// Function: sub_E409B0
// Address: 0xe409b0
//
__int64 __fastcall sub_E409B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  _BOOL4 v6; // r14d
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rbx
  _BYTE *v10; // rax
  _BYTE *v11; // r9
  __int64 result; // rax
  unsigned int v13; // esi
  __int64 v14; // r9
  int v15; // r11d
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rcx
  _DWORD *v20; // rdx
  int v21; // eax
  unsigned __int8 v22; // r8
  __int16 v23; // cx
  _BYTE *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rax
  _QWORD *v27; // r14
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r14
  __int64 v33; // r15
  unsigned int v34; // ebx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  int v38; // eax
  int v39; // ecx
  unsigned __int64 v40; // r12
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  int v44; // eax
  int v45; // edi
  __int64 v46; // rsi
  unsigned int v47; // eax
  __int64 v48; // r9
  int v49; // r11d
  __int64 *v50; // r10
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdi
  __int64 *v54; // r9
  int v55; // r10d
  unsigned int v56; // r8d
  __int64 v57; // rsi
  unsigned int v58; // [rsp+Ch] [rbp-84h]
  __int64 v59; // [rsp+18h] [rbp-78h]
  _BYTE *v60; // [rsp+18h] [rbp-78h]
  unsigned __int8 v61; // [rsp+20h] [rbp-70h]
  _BYTE *v62; // [rsp+20h] [rbp-70h]
  __int64 v63; // [rsp+20h] [rbp-70h]
  char v64; // [rsp+20h] [rbp-70h]
  _BYTE *v65; // [rsp+20h] [rbp-70h]
  const char *v66; // [rsp+28h] [rbp-68h]
  __int64 v67; // [rsp+28h] [rbp-68h]
  unsigned int v68; // [rsp+28h] [rbp-68h]
  unsigned int v69; // [rsp+28h] [rbp-68h]
  unsigned __int64 v70; // [rsp+30h] [rbp-60h] BYREF
  __int64 v71; // [rsp+38h] [rbp-58h]
  int v72; // [rsp+40h] [rbp-50h]
  __int16 v73; // [rsp+50h] [rbp-40h]

  v4 = a2;
  v6 = (*(_BYTE *)(a3 + 32) & 0xF) == 0;
  v7 = sub_B2F730(a3);
  if ( (*(_BYTE *)(a3 + 7) & 0x10) == 0 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v16 = 0;
      v17 = (v13 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = v14 + 16 * v17;
      v19 = *(_QWORD *)v18;
      if ( *(_QWORD *)v18 == a3 )
      {
LABEL_12:
        v20 = (_DWORD *)(v18 + 8);
        v21 = *(_DWORD *)(v18 + 8);
        if ( v21 )
        {
LABEL_13:
          v72 = v21;
          v70 = (unsigned __int64)"__unnamed_";
          v73 = 2307;
          switch ( *(_DWORD *)(v7 + 24) )
          {
            case 0:
            case 1:
            case 3:
            case 5:
            case 6:
            case 7:
              v22 = 0;
              return sub_E401D0(v4, (char *)&v70, v6, v7, v22);
            case 2:
            case 4:
              v22 = 95;
              return sub_E401D0(v4, (char *)&v70, v6, v7, v22);
            default:
              goto LABEL_96;
          }
        }
LABEL_53:
        v21 = *(_DWORD *)(a1 + 16);
        *v20 = v21;
        goto LABEL_13;
      }
      while ( v19 != -4096 )
      {
        if ( !v16 && v19 == -8192 )
          v16 = (__int64 *)v18;
        LODWORD(v17) = (v13 - 1) & (v15 + v17);
        v18 = v14 + 16LL * (unsigned int)v17;
        v19 = *(_QWORD *)v18;
        if ( *(_QWORD *)v18 == a3 )
          goto LABEL_12;
        ++v15;
      }
      if ( !v16 )
        v16 = (__int64 *)v18;
      v38 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v39 = v38 + 1;
      if ( 4 * (v38 + 1) < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(a1 + 20) - v39 > v13 >> 3 )
        {
LABEL_50:
          *(_DWORD *)(a1 + 16) = v39;
          if ( *v16 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v16 = a3;
          v20 = v16 + 1;
          *v20 = 0;
          goto LABEL_53;
        }
        v69 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
        sub_E407D0(a1, v13);
        v51 = *(_DWORD *)(a1 + 24);
        if ( v51 )
        {
          v52 = v51 - 1;
          v53 = *(_QWORD *)(a1 + 8);
          v54 = 0;
          v55 = 1;
          v56 = v52 & v69;
          v39 = *(_DWORD *)(a1 + 16) + 1;
          v16 = (__int64 *)(v53 + 16LL * (v52 & v69));
          v57 = *v16;
          if ( *v16 != a3 )
          {
            while ( v57 != -4096 )
            {
              if ( v57 == -8192 && !v54 )
                v54 = v16;
              v56 = v52 & (v55 + v56);
              v16 = (__int64 *)(v53 + 16LL * v56);
              v57 = *v16;
              if ( *v16 == a3 )
                goto LABEL_50;
              ++v55;
            }
            if ( v54 )
              v16 = v54;
          }
          goto LABEL_50;
        }
LABEL_95:
        ++*(_DWORD *)(a1 + 16);
LABEL_96:
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_E407D0(a1, 2 * v13);
    v44 = *(_DWORD *)(a1 + 24);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 8);
      v47 = (v44 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v39 = *(_DWORD *)(a1 + 16) + 1;
      v16 = (__int64 *)(v46 + 16LL * v47);
      v48 = *v16;
      if ( *v16 != a3 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -4096 )
        {
          if ( !v50 && v48 == -8192 )
            v50 = v16;
          v47 = v45 & (v49 + v47);
          v16 = (__int64 *)(v46 + 16LL * v47);
          v48 = *v16;
          if ( *v16 == a3 )
            goto LABEL_50;
          ++v49;
        }
        if ( v50 )
          v16 = v50;
      }
      goto LABEL_50;
    }
    goto LABEL_95;
  }
  v66 = sub_BD5D20(a3);
  v9 = v8;
  switch ( *(_DWORD *)(v7 + 24) )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
      v61 = 0;
      goto LABEL_4;
    case 2:
    case 4:
      v61 = 95;
LABEL_4:
      v10 = (_BYTE *)sub_B32590(a3);
      v11 = v10;
      if ( v10 && *v10 )
        v11 = 0;
      if ( v9 && (*v66 == 1 || (unsigned int)(*(_DWORD *)(v7 + 24) - 3) <= 1 && *v66 == 63) || !v11 )
        goto LABEL_9;
      v23 = (*((_WORD *)v11 + 1) >> 4) & 0x3FF;
      if ( *(_DWORD *)(v7 + 24) != 4 )
      {
        if ( v23 != 80 )
        {
LABEL_9:
          v71 = v9;
          v73 = 261;
          v70 = (unsigned __int64)v66;
          return sub_E401D0(a2, (char *)&v70, v6, v7, v61);
        }
LABEL_21:
        v73 = 261;
        v70 = (unsigned __int64)v66;
        v62 = v11;
        v71 = v9;
        sub_E401D0(a2, (char *)&v70, v6, v7, 0);
        v24 = *(_BYTE **)(a2 + 32);
        v25 = (__int64)v62;
        if ( (unsigned __int64)v24 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 64);
          v25 = (__int64)v62;
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v24 + 1;
          *v24 = 64;
        }
        goto LABEL_23;
      }
      if ( v23 == 65 )
      {
        v73 = 261;
        v65 = v11;
        v70 = (unsigned __int64)v66;
        v71 = v9;
        sub_E401D0(a2, (char *)&v70, v6, v7, 0x40u);
        v25 = (__int64)v65;
        goto LABEL_23;
      }
      if ( ((*((_WORD *)v11 + 1) >> 4) & 0x3FF) == 0x50 )
        goto LABEL_21;
      v58 = (*((_WORD *)v11 + 1) >> 4) & 0x3FF;
      v60 = v11;
      v73 = 261;
      v70 = (unsigned __int64)v66;
      v71 = v9;
      sub_E401D0(a2, (char *)&v70, v6, v7, v61);
      result = v58;
      if ( v58 > 0x41 )
        return result;
      v25 = (__int64)v60;
      if ( v58 <= 0x3F )
        return result;
LABEL_23:
      v26 = *(_QWORD *)(v25 + 24);
      if ( !(*(_DWORD *)(v26 + 8) >> 8)
        || (result = *(unsigned int *)(v26 + 12), (_DWORD)result == 1)
        || (_DWORD)result == 2
        && ((v27 = (_QWORD *)(v25 + 120), v67 = v25, v28 = sub_A74710((_QWORD *)(v25 + 120), 1, 85), v25 = v67, v28)
         || (result = sub_A74710(v27, 2, 85), v25 = v67, (_BYTE)result)) )
      {
        v63 = v25;
        v68 = sub_AE4380(v7, 0);
        if ( (*(_BYTE *)(v63 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v63, 0, v29, v30);
          v31 = *(_QWORD *)(v63 + 96);
          v32 = v31 + 40LL * *(_QWORD *)(v63 + 104);
          if ( (*(_BYTE *)(v63 + 2) & 1) != 0 )
          {
            sub_B2C6D0(v63, 0, v42, v43);
            v31 = *(_QWORD *)(v63 + 96);
          }
        }
        else
        {
          v31 = *(_QWORD *)(v63 + 96);
          v32 = v31 + 40LL * *(_QWORD *)(v63 + 104);
        }
        if ( v32 == v31 )
        {
          v40 = 0;
        }
        else
        {
          v33 = v31;
          v34 = 0;
          do
          {
            if ( !(unsigned __int8)sub_B2D720(v33) )
            {
              if ( (unsigned __int8)sub_B2BAE0(v33) )
              {
                v35 = sub_B2BC30(v33, v7);
              }
              else
              {
                v59 = *(_QWORD *)(v33 + 8);
                v64 = sub_AE5020(v7, v59);
                v36 = sub_9208B0(v7, v59);
                v71 = v37;
                v70 = ((1LL << v64) + ((unsigned __int64)(v36 + 7) >> 3) - 1) >> v64 << v64;
                v35 = sub_CA1930(&v70);
              }
              v34 += v68 * ((v35 != 0) + (unsigned int)((v35 - (unsigned __int64)(v35 != 0)) / v68));
            }
            v33 += 40;
          }
          while ( v32 != v33 );
          v40 = v34;
        }
        v41 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v41 >= *(_QWORD *)(a2 + 24) )
        {
          v4 = sub_CB5D20(a2, 64);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v41 + 1;
          *v41 = 64;
        }
        return sub_CB59D0(v4, v40);
      }
      return result;
    default:
      goto LABEL_96;
  }
}
