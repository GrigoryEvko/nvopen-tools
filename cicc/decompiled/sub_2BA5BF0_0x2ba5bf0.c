// Function: sub_2BA5BF0
// Address: 0x2ba5bf0
//
__int64 __fastcall sub_2BA5BF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char **a5)
{
  __int64 v6; // rcx
  __int64 v8; // rbx
  __int64 v9; // rax
  char v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // r8d
  __int64 v16; // rax
  int v17; // ecx
  int v18; // edx
  __int64 v19; // rax
  __int64 *v20; // r15
  __int64 v21; // r14
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rsi
  int v25; // ecx
  int v26; // ecx
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // rsi
  __int64 v33; // rdi
  _QWORD *j; // rax
  __int64 i; // rdx
  __int64 v36; // rcx
  __int64 v37; // rcx
  unsigned int v38; // ecx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  char v43; // r14
  __int64 *v44; // r12
  __int64 v45; // r15
  __int64 v46; // r8
  int v47; // edi
  __int64 v48; // rsi
  int v49; // edi
  unsigned int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // r8
  int v53; // eax
  __int64 v54; // rdi
  int v55; // edx
  unsigned int v56; // eax
  __int64 *v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // rax
  _QWORD *v60; // rdi
  __int64 v61; // rsi
  _QWORD *v62; // rax
  int v63; // r9d
  __int64 v64; // rcx
  unsigned int v65; // ecx
  __int64 v66; // rcx
  unsigned int v67; // ecx
  __int64 v68; // rcx
  unsigned int v69; // ecx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // r10
  bool v74; // dl
  __int64 v75; // rax
  int v76; // eax
  int v77; // ecx
  int v78; // r9d
  int v79; // eax
  int v80; // edx
  __int64 v81; // [rsp+10h] [rbp-A0h]
  __int64 *v85; // [rsp+38h] [rbp-78h]
  __int64 v86; // [rsp+50h] [rbp-60h] BYREF
  __int64 v87; // [rsp+58h] [rbp-58h]
  __int64 v88; // [rsp+60h] [rbp-50h] BYREF
  __int64 v89; // [rsp+68h] [rbp-48h]
  __int64 v90; // [rsp+70h] [rbp-40h]

  if ( **a5 == 84 || (unsigned __int8)sub_2B15E10(*a5, (__int64)a2, a3, a4, (unsigned int)a5) )
    goto LABEL_3;
  v8 = v6;
  if ( !a3 )
  {
    v9 = *(_QWORD *)(a1 + 168);
    v88 = a1;
    v90 = v6;
    v89 = v9;
    goto LABEL_7;
  }
  if ( sub_2B0D880(a2, a3, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B099C0)
    || sub_2B0D880(a2, a3, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B16010) )
  {
LABEL_3:
    v86 = 0;
    LOBYTE(v87) = 1;
    return v86;
  }
  v19 = *(_QWORD *)(a1 + 168);
  v88 = a1;
  v90 = v8;
  v89 = v19;
  v85 = &a2[a3];
  if ( v85 == a2 )
  {
LABEL_7:
    v10 = 0;
    goto LABEL_8;
  }
  v20 = a2;
  do
  {
    v21 = *v20;
    if ( (unsigned __int8)sub_2B14730(*v20) )
      goto LABEL_61;
    if ( *(_BYTE *)v21 <= 0x1Cu )
      BUG();
    v24 = *(_QWORD *)(v21 + 40);
    if ( *(_QWORD *)a1 == v24 )
    {
      v25 = *(_DWORD *)(a1 + 104);
      v22 = *(_QWORD *)(a1 + 88);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = v26 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v28 = (__int64 *)(v22 + 16LL * v27);
        v29 = *v28;
        if ( v21 == *v28 )
        {
LABEL_22:
          v30 = v28[1];
          if ( v30 && *(_DWORD *)(v30 + 136) == *(_DWORD *)(a1 + 204) )
            goto LABEL_61;
        }
        else
        {
          v76 = 1;
          while ( v29 != -4096 )
          {
            v23 = (unsigned int)(v76 + 1);
            v27 = v26 & (v76 + v27);
            v28 = (__int64 *)(v22 + 16LL * v27);
            v29 = *v28;
            if ( v21 == *v28 )
              goto LABEL_22;
            v76 = v23;
          }
        }
      }
    }
    v31 = *(_QWORD *)(a1 + 160);
    if ( v31 )
    {
      v32 = *(_QWORD *)(a1 + 168);
      v33 = *(_QWORD *)a1 + 48LL;
      j = (_QWORD *)(*(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      for ( i = v32 + 24; (_QWORD *)v33 != j; j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( !j )
          BUG();
        if ( *((_BYTE *)j - 24) != 85 )
          break;
        v68 = *(j - 7);
        if ( !v68 )
          break;
        if ( *(_BYTE *)v68 )
          break;
        v22 = j[7];
        if ( *(_QWORD *)(v68 + 24) != v22 || (*(_BYTE *)(v68 + 33) & 0x20) == 0 )
          break;
        v69 = *(_DWORD *)(v68 + 36);
        if ( v69 > 0xD3 )
        {
          if ( v69 != 324 )
          {
            if ( v69 > 0x144 )
            {
              if ( v69 != 376 )
                break;
            }
            else if ( v69 != 282 && v69 - 291 > 1 )
            {
              break;
            }
          }
        }
        else if ( v69 > 0x9A )
        {
          v22 = 1LL << ((unsigned __int8)v69 + 101);
          if ( (v22 & 0x186000000000001LL) == 0 )
            break;
        }
        else if ( v69 != 11 && v69 - 68 > 3 )
        {
          break;
        }
      }
      if ( v33 == i )
      {
LABEL_52:
        if ( (_QWORD *)v33 != j )
          goto LABEL_150;
      }
      else
      {
        while ( *(_BYTE *)(i - 24) == 85 )
        {
          v66 = *(_QWORD *)(i - 56);
          if ( !v66 )
            break;
          if ( *(_BYTE *)v66 )
            break;
          v22 = *(_QWORD *)(i + 56);
          if ( *(_QWORD *)(v66 + 24) != v22 || (*(_BYTE *)(v66 + 33) & 0x20) == 0 )
            break;
          v67 = *(_DWORD *)(v66 + 36);
          if ( v67 > 0xD3 )
          {
            if ( v67 != 324 )
            {
              if ( v67 > 0x144 )
              {
                if ( v67 != 376 )
                  break;
              }
              else if ( v67 != 282 && v67 - 291 > 1 )
              {
                break;
              }
            }
          }
          else if ( v67 > 0x9A )
          {
            v22 = 1LL << ((unsigned __int8)v67 + 101);
            if ( (v22 & 0x186000000000001LL) == 0 )
              break;
          }
          else if ( v67 != 11 && v67 - 68 > 3 )
          {
            break;
          }
          i = *(_QWORD *)(i + 8);
          if ( v33 == i )
            goto LABEL_52;
          if ( !i )
            BUG();
        }
LABEL_30:
        while ( (_QWORD *)v33 != j )
        {
          if ( v33 == i || j && (_QWORD *)v21 == j - 3 )
            goto LABEL_150;
          if ( i && v21 == i - 24 )
          {
            if ( v33 != i && (!j || (_QWORD *)v21 != j - 3) )
              goto LABEL_54;
            goto LABEL_150;
          }
          v36 = (unsigned int)(*(_DWORD *)(a1 + 196) + 1);
          *(_DWORD *)(a1 + 196) = v36;
          if ( (int)v36 > *(_DWORD *)(a1 + 200) )
          {
            sub_2BA57E0(&v88, 0, 0, v36, v22, v23);
            LOBYTE(v87) = 0;
            return v86;
          }
          i = *(_QWORD *)(i + 8);
          for ( j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL); (_QWORD *)v33 != j; j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL) )
          {
            if ( !j )
              BUG();
            if ( *((_BYTE *)j - 24) != 85 )
              break;
            v64 = *(j - 7);
            if ( !v64 )
              break;
            if ( *(_BYTE *)v64 )
              break;
            v22 = j[7];
            if ( *(_QWORD *)(v64 + 24) != v22 || (*(_BYTE *)(v64 + 33) & 0x20) == 0 )
              break;
            v65 = *(_DWORD *)(v64 + 36);
            if ( v65 > 0xD3 )
            {
              if ( v65 != 324 )
              {
                if ( v65 > 0x144 )
                {
                  if ( v65 != 376 )
                    break;
                }
                else if ( v65 != 282 && v65 - 291 > 1 )
                {
                  break;
                }
              }
            }
            else if ( v65 > 0x9A )
            {
              if ( ((1LL << ((unsigned __int8)v65 + 101)) & 0x186000000000001LL) == 0 )
                break;
            }
            else if ( v65 != 11 && v65 - 68 > 3 )
            {
              break;
            }
          }
          if ( v33 == i )
            goto LABEL_52;
          while ( 1 )
          {
            if ( !i )
              BUG();
            if ( *(_BYTE *)(i - 24) != 85 )
              break;
            v37 = *(_QWORD *)(i - 56);
            if ( !v37 )
              break;
            if ( *(_BYTE *)v37 )
              break;
            v22 = *(_QWORD *)(i + 56);
            if ( *(_QWORD *)(v37 + 24) != v22 || (*(_BYTE *)(v37 + 33) & 0x20) == 0 )
              break;
            v38 = *(_DWORD *)(v37 + 36);
            if ( v38 > 0xD3 )
            {
              if ( v38 != 324 )
              {
                if ( v38 > 0x144 )
                {
                  if ( v38 != 376 )
                    goto LABEL_30;
                }
                else if ( v38 != 282 && v38 - 291 > 1 )
                {
                  goto LABEL_30;
                }
              }
            }
            else if ( v38 > 0x9A )
            {
              v23 = 1LL << ((unsigned __int8)v38 + 101);
              if ( (v23 & 0x186000000000001LL) == 0 )
                goto LABEL_30;
            }
            else if ( v38 != 11 && v38 - 68 > 3 )
            {
              goto LABEL_30;
            }
            i = *(_QWORD *)(i + 8);
            if ( v33 == i )
              goto LABEL_52;
          }
        }
      }
      if ( v33 == i )
      {
LABEL_150:
        sub_2B5A300(a1, v21, v31, 0, *(_QWORD *)(a1 + 176));
        *(_QWORD *)(a1 + 160) = v21;
        goto LABEL_61;
      }
LABEL_54:
      v39 = *(_QWORD *)(v21 + 32);
      if ( v39 == *(_QWORD *)(v21 + 40) + 48LL || !v39 )
        v40 = 0;
      else
        v40 = v39 - 24;
      sub_2B5A300(a1, v32, v40, *(_QWORD *)(a1 + 184), 0);
      v41 = *(_QWORD *)(v21 + 32);
      if ( v41 == *(_QWORD *)(v21 + 40) + 48LL || !v41 )
        v42 = 0;
      else
        v42 = v41 - 24;
      *(_QWORD *)(a1 + 168) = v42;
    }
    else
    {
      v70 = *(_QWORD *)(v21 + 32);
      if ( v70 == v24 + 48 || !v70 )
        v71 = 0;
      else
        v71 = v70 - 24;
      v81 = *(_QWORD *)(a1 + 160);
      sub_2B5A300(a1, v21, v71, 0, 0);
      *(_QWORD *)(a1 + 160) = v21;
      v72 = *(_QWORD *)(v21 + 32);
      v73 = v81;
      if ( v72 != *(_QWORD *)(v21 + 40) + 48LL )
      {
        v74 = v72 != 0;
        v75 = v72 - 24;
        if ( v74 )
          v73 = v75;
      }
      *(_QWORD *)(a1 + 168) = v73;
    }
LABEL_61:
    ++v20;
  }
  while ( v85 != v20 );
  v43 = 0;
  v44 = a2;
  do
  {
    v45 = *v44;
    if ( !(unsigned __int8)sub_2B14730(*v44) )
    {
      v46 = 0;
      if ( *(_BYTE *)v45 > 0x1Cu && *(_QWORD *)a1 == *(_QWORD *)(v45 + 40) )
      {
        v47 = *(_DWORD *)(a1 + 104);
        v48 = *(_QWORD *)(a1 + 88);
        if ( v47 )
        {
          v49 = v47 - 1;
          v50 = v49 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v51 = (__int64 *)(v48 + 16LL * v50);
          v52 = *v51;
          if ( v45 == *v51 )
          {
LABEL_68:
            v46 = v51[1];
            if ( v46 && *(_DWORD *)(v46 + 136) != *(_DWORD *)(a1 + 204) )
              v46 = 0;
          }
          else
          {
            v79 = 1;
            while ( v52 != -4096 )
            {
              v80 = v79 + 1;
              v50 = v49 & (v79 + v50);
              v51 = (__int64 *)(v48 + 16LL * v50);
              v52 = *v51;
              if ( v45 == *v51 )
                goto LABEL_68;
              v79 = v80;
            }
            v46 = 0;
          }
        }
      }
      v53 = *(_DWORD *)(a1 + 136);
      v54 = *(_QWORD *)(a1 + 120);
      v86 = v46;
      if ( v53 )
      {
        v55 = v53 - 1;
        v56 = (v53 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v57 = (__int64 *)(v54 + 8LL * v56);
        v58 = *v57;
        if ( *v57 == v46 )
        {
LABEL_73:
          *v57 = -8192;
          v59 = *(unsigned int *)(a1 + 152);
          --*(_DWORD *)(a1 + 128);
          v60 = *(_QWORD **)(a1 + 144);
          ++*(_DWORD *)(a1 + 132);
          v61 = (__int64)&v60[v59];
          v62 = sub_2B0B870(v60, v61, &v86);
          if ( v62 + 1 != (_QWORD *)v61 )
          {
            memmove(v62, v62 + 1, v61 - (_QWORD)(v62 + 1));
            v63 = *(_DWORD *)(a1 + 152);
            v46 = v86;
          }
          *(_DWORD *)(a1 + 152) = v63 - 1;
        }
        else
        {
          v77 = 1;
          while ( v58 != -4096 )
          {
            v78 = v77 + 1;
            v56 = v55 & (v77 + v56);
            v57 = (__int64 *)(v54 + 8LL * v56);
            v58 = *v57;
            if ( v46 == *v57 )
              goto LABEL_73;
            v77 = v78;
          }
        }
      }
      if ( *(_BYTE *)(v46 + 152) )
        v43 = *(_BYTE *)(v46 + 152);
    }
    ++v44;
  }
  while ( v85 != v44 );
  v10 = v43;
LABEL_8:
  v11 = sub_2B2EF80(a1, a2, a3);
  sub_2BA57E0(&v88, v10, v11, v12, v13, v14);
  if ( v11 )
  {
    v16 = v11;
    v17 = 0;
    while ( 1 )
    {
      v18 = *(_DWORD *)(v16 + 148);
      if ( v18 == -1 )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      v17 += v18;
      if ( !v16 )
      {
        if ( v17 )
          break;
        goto LABEL_162;
      }
    }
LABEL_12:
    sub_2BA4B90(a1, a2, a3, *a5, v15);
    LOBYTE(v87) = 0;
  }
  else
  {
LABEL_162:
    if ( *(_BYTE *)(v11 + 152) )
      goto LABEL_12;
    v86 = v11;
    LOBYTE(v87) = 1;
  }
  return v86;
}
