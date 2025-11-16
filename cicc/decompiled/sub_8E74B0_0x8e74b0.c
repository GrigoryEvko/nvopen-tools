// Function: sub_8E74B0
// Address: 0x8e74b0
//
unsigned __int8 *__fastcall sub_8E74B0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v3; // r12
  unsigned __int8 v4; // al
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // r15
  unsigned __int8 *v9; // r14
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // ebx
  int v14; // eax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  _BYTE *v20; // r15
  _BYTE *v21; // rdi
  _BYTE *v22; // r12
  __int64 v23; // rax
  int v24; // ecx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rcx
  char v33; // al
  int v34; // r15d
  int v35; // ebx
  unsigned __int8 *v36; // r12
  unsigned __int8 *v37; // r14
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  unsigned __int8 v44; // al
  int v45; // eax
  __int64 v46; // rcx
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdx
  int v56; // r15d
  _BYTE *v57; // r12
  __int64 v58; // rax
  __int64 v59; // rbx
  unsigned __int8 *v60; // rdi
  __int64 v61; // rdx
  unsigned __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rcx
  unsigned __int8 *v66; // rax
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rcx
  _BYTE *v70; // r12
  __int64 v71; // r9
  __int64 v72; // rcx
  unsigned __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // rax
  unsigned __int64 v77; // rdx
  unsigned __int8 v78; // al
  unsigned __int8 v79; // al
  __int64 v80; // rax
  __int64 v81; // rcx
  unsigned __int64 v82; // rdx
  unsigned __int8 *v83; // rdi
  __int64 v84; // rcx
  unsigned __int64 v85; // rdx
  __int64 v86; // rax
  bool v87; // cc
  int v88; // ebx
  int v89; // eax
  __int64 v90; // rcx
  unsigned __int64 v91; // rax
  __int64 v92; // rbx
  int v93; // eax
  bool v94; // zf
  __int64 v95; // r12
  __int64 v96; // rdx
  unsigned __int64 v97; // rax
  __int64 v98; // r9
  __int64 v99; // rdx
  unsigned __int64 v100; // rax
  __int64 v101; // r9
  int v102; // [rsp+0h] [rbp-40h] BYREF
  int v103; // [rsp+4h] [rbp-3Ch] BYREF
  unsigned __int8 *v104[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1;
  v4 = *a1;
  if ( *a1 == 76 )
    return (unsigned __int8 *)sub_8ECF80(v3, a2);
  while ( 1 )
  {
    if ( v4 == 84 )
      return sub_8E5C30((__int64)v3, a2);
    if ( v4 == 102 )
    {
      v33 = v3[1];
      if ( v33 != 112 )
      {
        if ( v33 != 76 )
        {
          if ( v33 == 108 )
          {
            v34 = 1;
            v35 = 1;
          }
          else
          {
            v34 = 0;
            if ( v33 > 108 )
            {
              v35 = 1;
              if ( v33 == 114 )
                goto LABEL_63;
LABEL_83:
              v9 = v3;
              if ( *(_DWORD *)(a2 + 24) )
                return v9;
LABEL_118:
              ++*(_QWORD *)(a2 + 32);
              ++*(_QWORD *)(a2 + 48);
              *(_DWORD *)(a2 + 24) = 1;
              return v9;
            }
            v35 = 0;
            if ( v33 != 82 )
              goto LABEL_83;
          }
LABEL_63:
          v9 = v3 + 2;
          v36 = (unsigned __int8 *)sub_8E6070(v3 + 2, &v102, &v103, v104, a2);
          if ( !v36 )
            goto LABEL_117;
          v37 = &v9[v103];
          if ( !*(_QWORD *)(a2 + 32) )
          {
            v38 = *(_QWORD *)(a2 + 8);
            v39 = v38 + 1;
            if ( !*(_DWORD *)(a2 + 28) )
            {
              v40 = *(_QWORD *)(a2 + 16);
              if ( v39 < v40 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v38) = 40;
                v39 = *(_QWORD *)(a2 + 8) + 1LL;
              }
              else
              {
                *(_DWORD *)(a2 + 28) = 1;
                if ( v40 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v40 - 1) = 0;
                  v39 = *(_QWORD *)(a2 + 8) + 1LL;
                }
              }
            }
            *(_QWORD *)(a2 + 8) = v39;
          }
          if ( v35 )
          {
            if ( v34 )
            {
              if ( !*(_QWORD *)(a2 + 32) )
              {
                sub_8E5790((unsigned __int8 *)"...", a2);
                if ( !*(_QWORD *)(a2 + 32) )
                  sub_8E5790(v36, a2);
              }
              v9 = (unsigned __int8 *)sub_8E74B0(v37, a2);
            }
            else
            {
              v9 = (unsigned __int8 *)sub_8E74B0(v37, a2);
              if ( *(_QWORD *)(a2 + 32) )
                return v9;
              sub_8E5790(v36, a2);
              if ( *(_QWORD *)(a2 + 32) )
                return v9;
              sub_8E5790((unsigned __int8 *)"...", a2);
            }
          }
          else
          {
            v71 = sub_8E74B0(v37, a2);
            if ( !*(_QWORD *)(a2 + 32) )
            {
              sub_8E5790(v36, a2);
              if ( !*(_QWORD *)(a2 + 32) )
              {
                sub_8E5790((unsigned __int8 *)"...", a2);
                if ( !*(_QWORD *)(a2 + 32) )
                  sub_8E5790(v36, a2);
              }
            }
            v9 = (unsigned __int8 *)sub_8E74B0(v71, a2);
          }
          if ( *(_QWORD *)(a2 + 32) )
            return v9;
          v41 = *(_QWORD *)(a2 + 8);
          v42 = v41 + 1;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v43 = *(_QWORD *)(a2 + 16);
            if ( v43 > v42 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v41) = 41;
              v42 = *(_QWORD *)(a2 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a2 + 28) = 1;
              if ( v43 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v43 - 1) = 0;
                v42 = *(_QWORD *)(a2 + 8) + 1LL;
              }
            }
          }
LABEL_81:
          *(_QWORD *)(a2 + 8) = v42;
          return v9;
        }
        if ( (unsigned int)v3[2] - 48 > 9 )
        {
          v34 = 1;
          v35 = 0;
          goto LABEL_63;
        }
      }
      return sub_8E7080((__int64)v3, a2);
    }
    if ( v4 != 99 )
      break;
    v6 = v3[1];
    if ( v6 == 108 )
    {
      v53 = sub_8E74B0(v3 + 2, a2);
      goto LABEL_116;
    }
    if ( v6 == 112 )
    {
      if ( !*(_QWORD *)(a2 + 32) )
      {
        v63 = *(_QWORD *)(a2 + 8);
        v64 = v63 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v65 = *(_QWORD *)(a2 + 16);
          if ( v65 > v64 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v63) = 40;
            v64 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v65 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v65 - 1) = 0;
              v64 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v64;
      }
      v66 = sub_8E72C0(v3 + 2, 0, a2);
      v53 = (__int64)v66;
      if ( !*(_DWORD *)(a2 + 24) && *v66 == 73 )
        v53 = sub_8E9020(v66, a2);
      if ( !*(_QWORD *)(a2 + 32) )
      {
        v67 = *(_QWORD *)(a2 + 8);
        v68 = v67 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v69 = *(_QWORD *)(a2 + 16);
          if ( v68 < v69 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v67) = 41;
            v68 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v69 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v69 - 1) = 0;
              v68 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v68;
      }
      goto LABEL_116;
    }
    if ( v6 != 118 )
      goto LABEL_15;
    ++*(_QWORD *)(a2 + 32);
    v20 = v3 + 2;
    v21 = v3 + 2;
    v22 = (_BYTE *)sub_8E9FF0(v3 + 2, 0, 0, 0, 1, a2);
    sub_8EB260(v21, 0, 0, a2);
    LODWORD(v21) = *(_DWORD *)(a2 + 24);
    v23 = *(_QWORD *)(a2 + 32) - 1LL;
    *(_QWORD *)(a2 + 32) = v23;
    if ( (_DWORD)v21 || *v22 == 95 )
    {
      ++*(_QWORD *)(a2 + 48);
      v9 = (unsigned __int8 *)sub_8E9FF0(v20, 0, 0, 0, 1, a2);
      sub_8EB260(v20, 0, 0, a2);
      v29 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 48) = v29 - 1;
      if ( *(_DWORD *)(a2 + 24) )
        return v9;
      if ( *v9 != 95 )
      {
        ++*(_QWORD *)(a2 + 32);
        *(_DWORD *)(a2 + 24) = 1;
        *(_QWORD *)(a2 + 48) = v29;
        return v9;
      }
      v53 = (__int64)(v9 + 1);
LABEL_116:
      v9 = (unsigned __int8 *)sub_8E8A30(v53, 69, 40, 41, 0, a2);
      if ( *v9 != 69 )
      {
LABEL_117:
        if ( *(_DWORD *)(a2 + 24) )
          return v9;
        goto LABEL_118;
      }
      return ++v9;
    }
    if ( !v23 )
    {
      v30 = *(_QWORD *)(a2 + 8);
      v31 = v30 + 1;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v32 = *(_QWORD *)(a2 + 16);
        if ( v31 < v32 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v30) = 40;
          v31 = *(_QWORD *)(a2 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a2 + 28) = 1;
          if ( v32 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v32 - 1) = 0;
            v31 = *(_QWORD *)(a2 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a2 + 8) = v31;
    }
    ++*(_QWORD *)(a2 + 48);
    v3 = (unsigned __int8 *)sub_8E9FF0(v20, 0, 0, 0, 1, a2);
    sub_8EB260(v20, 0, 0, a2);
    v24 = *(_DWORD *)(a2 + 24);
    --*(_QWORD *)(a2 + 48);
    if ( v24 )
      return v3;
    if ( *(_QWORD *)(a2 + 32) )
    {
LABEL_8:
      v4 = *v3;
      if ( *v3 == 76 )
        return (unsigned __int8 *)sub_8ECF80(v3, a2);
    }
    else
    {
      v25 = *(_QWORD *)(a2 + 8);
      v26 = v25 + 1;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v27 = *(_QWORD *)(a2 + 16);
        if ( v27 > v26 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v25) = 41;
          v26 = *(_QWORD *)(a2 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a2 + 28) = 1;
          if ( v27 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v27 - 1) = 0;
            v26 = *(_QWORD *)(a2 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a2 + 8) = v26;
      v4 = *v3;
      if ( *v3 == 76 )
        return (unsigned __int8 *)sub_8ECF80(v3, a2);
    }
  }
  if ( v4 != 103 )
  {
    if ( v4 == 110 )
    {
      v44 = v3[1];
      if ( v44 == 119 )
      {
        if ( !*(_QWORD *)(a2 + 32) )
          sub_8E5790((unsigned __int8 *)"new ", a2);
      }
      else
      {
        if ( v44 != 97 )
          goto LABEL_15;
        if ( !*(_QWORD *)(a2 + 32) )
          sub_8E5790("new[] ", a2);
      }
      v9 = v3 + 2;
      if ( v3[2] == 95 )
      {
        v45 = *(_DWORD *)(a2 + 24);
      }
      else
      {
        v9 = (unsigned __int8 *)sub_8E8A30(v3 + 2, 95, 40, 41, 0, a2);
        if ( !*(_QWORD *)(a2 + 32) )
        {
          v75 = *(_QWORD *)(a2 + 8);
          v76 = v75 + 1;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v77 = *(_QWORD *)(a2 + 16);
            if ( v76 < v77 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v75) = 32;
              v76 = *(_QWORD *)(a2 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a2 + 28) = 1;
              if ( v77 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v77 - 1) = 0;
                v76 = *(_QWORD *)(a2 + 8) + 1LL;
              }
            }
          }
          *(_QWORD *)(a2 + 8) = v76;
        }
        v45 = *(_DWORD *)(a2 + 24);
        if ( *v9 != 95 )
        {
          if ( v45 )
            return v9;
          goto LABEL_118;
        }
      }
      v3 = v9 + 1;
      if ( v45 )
        return v3;
      goto LABEL_93;
    }
    if ( v4 == 100 )
    {
      if ( v3[1] != 116 )
        goto LABEL_15;
      if ( !*(_QWORD *)(a2 + 32) )
      {
        v46 = *(_QWORD *)(a2 + 8);
        v47 = v46 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v48 = *(_QWORD *)(a2 + 16);
          if ( v48 > v47 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v46) = 40;
            v47 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v48 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v48 - 1) = 0;
              v47 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v47;
      }
      v9 = (unsigned __int8 *)sub_8E74B0(v3 + 2, a2);
      if ( *(_DWORD *)(a2 + 24) )
        return v9;
      if ( !*(_QWORD *)(a2 + 32) )
      {
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v49 = *(_QWORD *)(a2 + 8);
          v50 = *(_QWORD *)(a2 + 16);
          if ( v49 + 1 < v50 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v49) = 46;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v50 )
              *(_BYTE *)(*(_QWORD *)a2 + v50 - 1) = 0;
          }
        }
        ++*(_QWORD *)(a2 + 8);
      }
      v9 = (unsigned __int8 *)sub_8ED760(v9, a2);
      if ( *(_QWORD *)(a2 + 32) )
        return v9;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v51 = *(_QWORD *)(a2 + 8);
        v52 = *(_QWORD *)(a2 + 16);
        if ( v51 + 1 >= v52 )
          goto LABEL_112;
LABEL_195:
        *(_BYTE *)(*(_QWORD *)a2 + v51) = 41;
      }
    }
    else
    {
      if ( v4 != 112 )
      {
        if ( v4 != 115 )
        {
          if ( v4 == 116 )
          {
            if ( v3[1] != 108 )
              goto LABEL_15;
            v70 = v3 + 2;
            v9 = (unsigned __int8 *)sub_8E9FF0(v70, 0, 0, 0, 1, a2);
            sub_8EB260(v70, 0, 0, a2);
          }
          else
          {
            if ( v4 != 105 )
              goto LABEL_15;
            v9 = v3 + 2;
            if ( v3[1] != 108 )
              goto LABEL_15;
          }
          if ( *(_DWORD *)(a2 + 24) )
            return v9;
          v9 = (unsigned __int8 *)sub_8E8A30(v9, 69, 123, 125, 1, a2);
          if ( *v9 != 69 )
            goto LABEL_117;
          return ++v9;
        }
        v78 = v3[1];
        if ( v78 == 90 )
        {
          if ( !*(_QWORD *)(a2 + 32) )
            sub_8E5790("sizeof...(", a2);
          v79 = v3[2];
          v9 = v3 + 2;
          if ( v79 == 84 )
          {
            v9 = sub_8E5C30((__int64)(v3 + 2), a2);
            v80 = *(_QWORD *)(a2 + 32);
          }
          else if ( v79 == 102 )
          {
            v9 = sub_8E7080((__int64)(v3 + 2), a2);
            v80 = *(_QWORD *)(a2 + 32);
          }
          else
          {
            v80 = *(_QWORD *)(a2 + 32);
            if ( !*(_DWORD *)(a2 + 24) )
            {
              ++v80;
              ++*(_QWORD *)(a2 + 48);
              *(_DWORD *)(a2 + 24) = 1;
              *(_QWORD *)(a2 + 32) = v80;
            }
          }
          if ( v80 )
            return v9;
          v81 = *(_QWORD *)(a2 + 8);
          v42 = v81 + 1;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v82 = *(_QWORD *)(a2 + 16);
            if ( v82 > v42 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v81) = 41;
              v42 = *(_QWORD *)(a2 + 8) + 1LL;
            }
            else
            {
              *(_DWORD *)(a2 + 28) = 1;
              if ( v82 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v82 - 1) = 0;
                v42 = *(_QWORD *)(a2 + 8) + 1LL;
              }
            }
          }
          goto LABEL_81;
        }
        if ( v78 == 112 )
        {
          v9 = (unsigned __int8 *)sub_8E74B0(v3 + 2, a2);
          if ( !*(_QWORD *)(a2 + 32) )
            sub_8E5790((unsigned __int8 *)"...", a2);
          return v9;
        }
LABEL_15:
        v7 = (unsigned __int8 *)sub_8E6070(v3, &v102, &v103, v104, a2);
        v8 = v7;
        if ( !v7 )
          return (unsigned __int8 *)sub_8ED760(v3, a2);
        v9 = &v3[v103];
        if ( !memcmp(v7, "builtin-operation-", 0x12u) )
        {
          if ( *(_QWORD *)(a2 + 32) )
            goto LABEL_24;
          sub_8E5790(v7, a2);
          if ( *(_QWORD *)(a2 + 32) )
            goto LABEL_24;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v10 = *(_QWORD *)(a2 + 8);
            v11 = *(_QWORD *)(a2 + 16);
            v12 = v10 + 1;
            if ( v10 + 1 >= v11 )
            {
              *(_DWORD *)(a2 + 28) = 1;
              if ( v11 )
              {
                *(_BYTE *)(*(_QWORD *)a2 + v11 - 1) = 0;
                v12 = *(_QWORD *)(a2 + 8) + 1LL;
              }
              goto LABEL_23;
            }
            *(_BYTE *)(*(_QWORD *)a2 + v10) = 40;
          }
          v12 = *(_QWORD *)(a2 + 8) + 1LL;
LABEL_23:
          *(_QWORD *)(a2 + 8) = v12;
LABEL_24:
          if ( v102 > 0 )
          {
            v13 = 1;
            do
            {
              if ( *v9 == 84 && v9[1] == 79 )
              {
                v83 = v9 + 2;
                v9 = (unsigned __int8 *)sub_8E9FF0(v9 + 2, 0, 0, 0, 1, a2);
                sub_8EB260(v83, 0, 0, a2);
              }
              else
              {
                v9 = (unsigned __int8 *)sub_8E74B0(v9, a2);
              }
              v14 = v102;
              if ( v102 != v13 && !*(_QWORD *)(a2 + 32) )
              {
                sub_8E5790((unsigned __int8 *)", ", a2);
                v14 = v102;
              }
              ++v13;
            }
            while ( v13 <= v14 );
          }
          if ( *(_QWORD *)(a2 + 32) )
            return v9;
          v15 = *(_QWORD *)(a2 + 8);
          v16 = v15 + 1;
          if ( !*(_DWORD *)(a2 + 28) )
          {
            v17 = *(_QWORD *)(a2 + 16);
            if ( v17 > v16 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v15) = 41;
              v18 = *(_QWORD *)(a2 + 8) + 1LL;
              v19 = *(_QWORD *)(a2 + 32);
              goto LABEL_38;
            }
            *(_DWORD *)(a2 + 28) = 1;
            if ( v17 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v17 - 1) = 0;
              v18 = *(_QWORD *)(a2 + 8) + 1LL;
              v19 = *(_QWORD *)(a2 + 32);
LABEL_38:
              *(_QWORD *)(a2 + 8) = v18;
              goto LABEL_164;
            }
          }
          goto LABEL_208;
        }
        if ( memcmp(v7, "subscript", 9u) )
        {
          if ( v102 != 1 )
          {
            if ( v102 == 2 )
            {
              v98 = sub_8E74B0(&v3[v103], a2);
              if ( !*(_QWORD *)(a2 + 32) )
                sub_8E5790(v8, a2);
            }
            else
            {
              if ( v102 != 3 )
              {
                if ( !strcmp((const char *)v7, "sizeof(")
                  || !strcmp((const char *)v7, "alignof(")
                  || !strcmp((const char *)v7, "__alignof__(") )
                {
                  if ( !*(_QWORD *)(a2 + 32) )
                    sub_8E5790(v7, a2);
                  v92 = sub_8E9FF0(v9, 0, 0, 0, 1, a2);
                  sub_8EB260(v9, 0, 0, a2);
                  v9 = (unsigned __int8 *)v92;
                  v19 = *(_QWORD *)(a2 + 32);
                }
                else if ( !strcmp((const char *)v7, "__uuidof(") || !strcmp((const char *)v7, "typeid(") )
                {
                  if ( !*(_QWORD *)(a2 + 32) )
                    sub_8E5790(v8, a2);
                  v9 = (unsigned __int8 *)sub_8EB9C0(v9, 1, 0, a2);
                  v19 = *(_QWORD *)(a2 + 32);
                }
                else
                {
                  if ( !strcmp((const char *)v8, "::typeid") )
                  {
                    v9 = (unsigned __int8 *)sub_8EB9C0(v9, 1, 0, a2);
                    if ( *(_QWORD *)(a2 + 32) )
                      return v9;
                    goto LABEL_163;
                  }
                  if ( !strcmp((const char *)v8, "throw") )
                  {
                    if ( *(_QWORD *)(a2 + 32) )
                      return v9;
                    goto LABEL_163;
                  }
                  if ( !strncmp((const char *)v8, "\"\"", 2u) )
                  {
                    if ( *(_QWORD *)(a2 + 32) )
                      return v9;
                    sub_8E5790("operator ", a2);
                    if ( *(_QWORD *)(a2 + 32) )
                      return v9;
LABEL_163:
                    sub_8E5790(v8, a2);
                    v19 = *(_QWORD *)(a2 + 32);
                    goto LABEL_164;
                  }
                  if ( !*(_DWORD *)(a2 + 24) )
                    sub_8E5770(a2);
                  v19 = *(_QWORD *)(a2 + 32);
                }
LABEL_164:
                if ( v19 )
                  return v9;
LABEL_165:
                sub_8E5790(v104[0], a2);
                return v9;
              }
              v101 = sub_8E74B0(&v3[v103], a2);
              if ( !*(_QWORD *)(a2 + 32) )
                sub_8E5790(v8, a2);
              v98 = sub_8E74B0(v101, a2);
              if ( !*(_QWORD *)(a2 + 32) )
                sub_8E5790((unsigned __int8 *)":", a2);
            }
            v9 = (unsigned __int8 *)sub_8E74B0(v98, a2);
            v19 = *(_QWORD *)(a2 + 32);
            goto LABEL_164;
          }
          v93 = *v7;
          if ( v93 == 43 && v8[1] == 43 && !v8[2] || v93 == 45 && v8[1] == 45 && !v8[2] )
          {
            if ( *v9 != 95 )
            {
              v94 = *(_QWORD *)(a2 + 32) == 0;
              v104[0] = v8;
              if ( !v94 )
              {
LABEL_257:
                v9 = (unsigned __int8 *)sub_8E74B0(v9, a2);
                v19 = *(_QWORD *)(a2 + 32);
                goto LABEL_164;
              }
              v8 = (unsigned __int8 *)byte_3F871B3;
              sub_8E5790((unsigned __int8 *)byte_3F871B3, a2);
              goto LABEL_305;
            }
            ++v9;
          }
          if ( *(_QWORD *)(a2 + 32) )
          {
            if ( !strcmp((const char *)v8, "static_cast") || !strcmp((const char *)v8, "dynamic_cast") )
              goto LABEL_273;
          }
          else
          {
            sub_8E5790(v8, a2);
            if ( !strcmp((const char *)v8, "static_cast") || !strcmp((const char *)v8, "dynamic_cast") )
            {
LABEL_285:
              if ( !*(_QWORD *)(a2 + 32) )
              {
                if ( !*(_DWORD *)(a2 + 28) )
                {
                  v99 = *(_QWORD *)(a2 + 8);
                  v100 = *(_QWORD *)(a2 + 16);
                  if ( v99 + 1 < v100 )
                  {
                    *(_BYTE *)(*(_QWORD *)a2 + v99) = 60;
                  }
                  else
                  {
                    *(_DWORD *)(a2 + 28) = 1;
                    if ( v100 )
                      *(_BYTE *)(*(_QWORD *)a2 + v100 - 1) = 0;
                  }
                }
                ++*(_QWORD *)(a2 + 8);
              }
LABEL_273:
              v95 = sub_8E9FF0(v9, 0, 0, 0, 1, a2);
              sub_8EB260(v9, 0, 0, a2);
              if ( !*(_QWORD *)(a2 + 32) )
                sub_8E5790(">(", a2);
              v9 = (unsigned __int8 *)sub_8E74B0(v95, a2);
              if ( *(_QWORD *)(a2 + 32) )
                return v9;
              if ( !*(_DWORD *)(a2 + 28) )
              {
                v96 = *(_QWORD *)(a2 + 8);
                v97 = *(_QWORD *)(a2 + 16);
                if ( v97 > v96 + 1 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v96) = 41;
                  v19 = *(_QWORD *)(a2 + 32);
                  goto LABEL_280;
                }
                *(_DWORD *)(a2 + 28) = 1;
                if ( v97 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v97 - 1) = 0;
                  v19 = *(_QWORD *)(a2 + 32);
LABEL_280:
                  ++*(_QWORD *)(a2 + 8);
                  goto LABEL_164;
                }
              }
              ++*(_QWORD *)(a2 + 8);
              goto LABEL_165;
            }
          }
LABEL_305:
          if ( strcmp((const char *)v8, "const_cast")
            && strcmp((const char *)v8, "reinterpret_cast")
            && strcmp((const char *)v8, "safe_cast") )
          {
            goto LABEL_257;
          }
          goto LABEL_285;
        }
        v9 = (unsigned __int8 *)sub_8E74B0(&v3[v103], a2);
        if ( *(_QWORD *)(a2 + 32) )
        {
          if ( v102 <= 1 )
            return v9;
          goto LABEL_234;
        }
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v84 = *(_QWORD *)(a2 + 8);
          v85 = *(_QWORD *)(a2 + 16);
          v86 = v84 + 1;
          if ( v84 + 1 >= v85 )
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v85 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v85 - 1) = 0;
              v86 = *(_QWORD *)(a2 + 8) + 1LL;
            }
LABEL_233:
            v87 = v102 <= 1;
            *(_QWORD *)(a2 + 8) = v86;
            if ( v87 )
            {
LABEL_239:
              if ( *(_QWORD *)(a2 + 32) )
                return v9;
              v90 = *(_QWORD *)(a2 + 8);
              v16 = v90 + 1;
              if ( !*(_DWORD *)(a2 + 28) )
              {
                v91 = *(_QWORD *)(a2 + 16);
                if ( v91 > v16 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v90) = 93;
                  v18 = *(_QWORD *)(a2 + 8) + 1LL;
                  v19 = *(_QWORD *)(a2 + 32);
                  goto LABEL_38;
                }
                *(_DWORD *)(a2 + 28) = 1;
                if ( v91 )
                {
                  *(_BYTE *)(*(_QWORD *)a2 + v91 - 1) = 0;
                  v18 = *(_QWORD *)(a2 + 8) + 1LL;
                  v19 = *(_QWORD *)(a2 + 32);
                  goto LABEL_38;
                }
              }
LABEL_208:
              *(_QWORD *)(a2 + 8) = v16;
              goto LABEL_165;
            }
LABEL_234:
            v88 = 2;
            do
            {
              v9 = (unsigned __int8 *)sub_8E74B0(v9, a2);
              v89 = v102;
              if ( v102 != v88 && !*(_QWORD *)(a2 + 32) )
              {
                sub_8E5790((unsigned __int8 *)", ", a2);
                v89 = v102;
              }
              ++v88;
            }
            while ( v88 <= v89 );
            goto LABEL_239;
          }
          *(_BYTE *)(*(_QWORD *)a2 + v84) = 91;
        }
        v86 = *(_QWORD *)(a2 + 8) + 1LL;
        goto LABEL_233;
      }
      if ( v3[1] != 116 )
        goto LABEL_15;
      if ( !*(_QWORD *)(a2 + 32) )
      {
        v72 = *(_QWORD *)(a2 + 8);
        v73 = v72 + 1;
        if ( !*(_DWORD *)(a2 + 28) )
        {
          v74 = *(_QWORD *)(a2 + 16);
          if ( v73 < v74 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v72) = 40;
            v73 = *(_QWORD *)(a2 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a2 + 28) = 1;
            if ( v74 )
            {
              *(_BYTE *)(*(_QWORD *)a2 + v74 - 1) = 0;
              v73 = *(_QWORD *)(a2 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a2 + 8) = v73;
      }
      v9 = (unsigned __int8 *)sub_8E74B0(v3 + 2, a2);
      if ( *(_DWORD *)(a2 + 24) )
        return v9;
      if ( !*(_QWORD *)(a2 + 32) )
        sub_8E5790((unsigned __int8 *)"->", a2);
      v9 = (unsigned __int8 *)sub_8ED760(v9, a2);
      if ( *(_QWORD *)(a2 + 32) )
        return v9;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v51 = *(_QWORD *)(a2 + 8);
        v52 = *(_QWORD *)(a2 + 16);
        if ( v52 <= v51 + 1 )
        {
LABEL_112:
          *(_DWORD *)(a2 + 28) = 1;
          if ( v52 )
            *(_BYTE *)(*(_QWORD *)a2 + v52 - 1) = 0;
          goto LABEL_114;
        }
        goto LABEL_195;
      }
    }
LABEL_114:
    ++*(_QWORD *)(a2 + 8);
    return v9;
  }
  v5 = v3[1];
  if ( v5 == 115 )
  {
    if ( !*(_QWORD *)(a2 + 32) )
      sub_8E5790((unsigned __int8 *)"::", a2);
    v3 += 2;
    goto LABEL_8;
  }
  if ( v5 != 99 )
    goto LABEL_15;
  if ( !*(_QWORD *)(a2 + 32) )
    sub_8E5790("gcnew ", a2);
  if ( v3[2] == 95 )
  {
    v3 += 3;
LABEL_93:
    v9 = (unsigned __int8 *)sub_8E9FF0(v3, 0, 0, 0, 1, a2);
    sub_8EB260(v3, 0, 0, a2);
    goto LABEL_132;
  }
  ++*(_QWORD *)(a2 + 32);
  v9 = v3 + 2;
  ++*(_QWORD *)(a2 + 48);
  v54 = sub_8E8A30(v3 + 2, 95, 40, 41, 0, a2);
  v55 = *(_QWORD *)(a2 + 32);
  v56 = *(_DWORD *)(a2 + 24);
  v57 = (_BYTE *)v54;
  *(_QWORD *)(a2 + 32) = v55 - 1;
  v58 = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a2 + 48) = v58 - 1;
  if ( v56 )
    return v9;
  if ( *v57 == 95 )
  {
    ++v57;
  }
  else
  {
    *(_DWORD *)(a2 + 24) = 1;
    *(_QWORD *)(a2 + 32) = v55;
    *(_QWORD *)(a2 + 48) = v58;
  }
  v59 = sub_8E9FF0(v57, 0, 0, 0, 1, a2);
  sub_8EB260(v57, 0, 0, a2);
  v60 = v9;
  v9 = (unsigned __int8 *)v59;
  sub_8E8A30(v60, 95, 40, 41, 0, a2);
  if ( !*(_QWORD *)(a2 + 32) )
  {
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v61 = *(_QWORD *)(a2 + 8);
      v62 = *(_QWORD *)(a2 + 16);
      if ( v61 + 1 < v62 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v61) = 32;
      }
      else
      {
        *(_DWORD *)(a2 + 28) = 1;
        if ( v62 )
          *(_BYTE *)(*(_QWORD *)a2 + v62 - 1) = 0;
      }
    }
    ++*(_QWORD *)(a2 + 8);
    v9 = (unsigned __int8 *)v59;
  }
LABEL_132:
  if ( *(_DWORD *)(a2 + 24) )
    return v9;
  return (unsigned __int8 *)sub_8E8E80(v9, a2);
}
