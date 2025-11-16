// Function: sub_26BE700
// Address: 0x26be700
//
void __fastcall sub_26BE700(char *a1, char *a2)
{
  _QWORD *v2; // r12
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rax
  int *v5; // r14
  size_t v6; // r13
  int *v7; // r14
  size_t v8; // r12
  char *j; // r15
  _QWORD *v10; // r14
  char v11; // bl
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // esi
  _QWORD *v18; // r12
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  unsigned int v23; // eax
  _QWORD *v24; // r9
  _QWORD *v25; // r12
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  unsigned int v30; // eax
  __int64 v31; // r8
  __int64 v32; // r15
  __int64 v33; // r12
  bool v34; // cf
  bool v35; // al
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // esi
  _QWORD *v40; // r14
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r15
  unsigned int v45; // eax
  _QWORD *v46; // r9
  _QWORD *v47; // r14
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r13
  unsigned int v52; // eax
  __int64 v53; // r8
  __int64 v54; // r13
  __int64 v55; // r14
  unsigned __int64 v56; // r13
  int *v57; // r13
  size_t v58; // r12
  int *v59; // rbx
  size_t v60; // r13
  _QWORD *v61; // [rsp+0h] [rbp-150h]
  unsigned __int64 v63; // [rsp+10h] [rbp-140h]
  _QWORD *v64; // [rsp+10h] [rbp-140h]
  _QWORD *v65; // [rsp+18h] [rbp-138h]
  unsigned __int64 v66; // [rsp+18h] [rbp-138h]
  unsigned __int64 v67; // [rsp+28h] [rbp-128h]
  _QWORD *v68; // [rsp+28h] [rbp-128h]
  _QWORD *v69; // [rsp+30h] [rbp-120h]
  _QWORD *v70; // [rsp+30h] [rbp-120h]
  char *v71; // [rsp+38h] [rbp-118h]
  char *v72; // [rsp+38h] [rbp-118h]
  _QWORD *v73; // [rsp+48h] [rbp-108h]
  _QWORD *v74; // [rsp+48h] [rbp-108h]
  _QWORD *v75; // [rsp+50h] [rbp-100h]
  _QWORD *v76; // [rsp+50h] [rbp-100h]
  char *v77; // [rsp+58h] [rbp-F8h]
  char *i; // [rsp+60h] [rbp-F0h]
  _QWORD *v79; // [rsp+68h] [rbp-E8h]
  _QWORD v80[2]; // [rsp+70h] [rbp-E0h] BYREF
  int v81[52]; // [rsp+80h] [rbp-D0h] BYREF

  if ( a1 != a2 && a1 + 8 != a2 )
  {
    for ( i = a1 + 8; a2 != i; i += 8 )
    {
      v2 = *(_QWORD **)a1;
      v79 = *(_QWORD **)i;
      v3 = sub_EF9210(*(_QWORD **)i);
      v4 = sub_EF9210(v2);
      if ( v3 == v4 )
      {
        v5 = (int *)v79[2];
        v6 = v79[3];
        if ( v5 )
        {
          sub_C7D030(v81);
          sub_C7D280(v81, v5, v6);
          sub_C7D290(v81, v80);
          v6 = v80[0];
        }
        v7 = (int *)v2[2];
        v8 = v2[3];
        if ( v7 )
        {
          sub_C7D030(v81);
          sub_C7D280(v81, v7, v8);
          sub_C7D290(v81, v80);
          v8 = v80[0];
        }
        v79 = *(_QWORD **)i;
        if ( v8 > v6 )
        {
LABEL_6:
          if ( a1 != i )
            memmove(a1 + 8, a1, i - a1);
          *(_QWORD *)a1 = v79;
          continue;
        }
      }
      else if ( v3 > v4 )
      {
        goto LABEL_6;
      }
      for ( j = i; ; *((_QWORD *)j + 1) = *(_QWORD *)j )
      {
        v10 = (_QWORD *)*((_QWORD *)j - 1);
        v77 = j;
        v11 = unk_4F838D3;
        if ( unk_4F838D3 )
        {
          v12 = v79[8];
          if ( v12 )
            goto LABEL_19;
        }
        v36 = v79[20];
        if ( !v79[14] )
        {
          if ( !v36 )
            goto LABEL_96;
          v38 = v79[18];
LABEL_59:
          v76 = (_QWORD *)(v38 + 48);
          if ( *(_QWORD *)(v38 + 64) == v38 + 48 )
          {
LABEL_96:
            v12 = v79[7] != 0;
            goto LABEL_86;
          }
          v72 = j;
          v12 = 0;
          v70 = (_QWORD *)*((_QWORD *)j - 1);
          v40 = *(_QWORD **)(v38 + 64);
          while ( 2 )
          {
            if ( v11 )
            {
              v41 = v40[14];
              if ( v41 )
              {
LABEL_83:
                v12 += v41;
                v40 = (_QWORD *)sub_220EF30((__int64)v40);
                if ( v76 == v40 )
                {
                  j = v72;
                  v10 = v70;
                  goto LABEL_85;
                }
                continue;
              }
            }
            break;
          }
          v42 = v40[26];
          if ( v40[20] )
          {
            v43 = v40[18];
            if ( !v42
              || (v44 = v40[24], v45 = *(_DWORD *)(v44 + 32), *(_DWORD *)(v43 + 32) < v45)
              || *(_DWORD *)(v43 + 32) == v45 && *(_DWORD *)(v43 + 36) < *(_DWORD *)(v44 + 36) )
            {
              v41 = *(_QWORD *)(v43 + 40);
LABEL_82:
              if ( v41 )
                goto LABEL_83;
LABEL_98:
              v41 = v40[13] != 0;
              goto LABEL_83;
            }
          }
          else
          {
            if ( !v42 )
              goto LABEL_98;
            v44 = v40[24];
          }
          v46 = *(_QWORD **)(v44 + 64);
          v74 = (_QWORD *)(v44 + 48);
          if ( v46 == (_QWORD *)(v44 + 48) )
            goto LABEL_98;
          v66 = v12;
          v41 = 0;
          v68 = v40;
          v47 = v46;
          while ( 2 )
          {
            if ( v11 )
            {
              v48 = v47[14];
              if ( v48 )
                goto LABEL_80;
            }
            v49 = v47[26];
            if ( v47[20] )
            {
              v50 = v47[18];
              if ( !v49
                || (v51 = v47[24], v52 = *(_DWORD *)(v51 + 32), *(_DWORD *)(v50 + 32) < v52)
                || *(_DWORD *)(v50 + 32) == v52 && *(_DWORD *)(v50 + 36) < *(_DWORD *)(v51 + 36) )
              {
                v48 = *(_QWORD *)(v50 + 40);
                goto LABEL_79;
              }
LABEL_75:
              v53 = *(_QWORD *)(v51 + 64);
              v54 = v51 + 48;
              if ( v53 != v54 )
              {
                v64 = v47;
                v48 = 0;
                v55 = v53;
                do
                {
                  v48 += sub_EF9210((_QWORD *)(v55 + 48));
                  v55 = sub_220EF30(v55);
                }
                while ( v54 != v55 );
                v47 = v64;
LABEL_79:
                if ( v48 )
                {
LABEL_80:
                  v41 += v48;
                  v47 = (_QWORD *)sub_220EF30((__int64)v47);
                  if ( v74 == v47 )
                  {
                    v40 = v68;
                    v12 = v66;
                    goto LABEL_82;
                  }
                  continue;
                }
              }
            }
            else if ( v49 )
            {
              v51 = v47[24];
              goto LABEL_75;
            }
            break;
          }
          v48 = v47[13] != 0;
          goto LABEL_80;
        }
        v37 = v79[12];
        if ( v36 )
        {
          v38 = v79[18];
          v39 = *(_DWORD *)(v38 + 32);
          if ( *(_DWORD *)(v37 + 32) >= v39
            && (*(_DWORD *)(v37 + 32) != v39 || *(_DWORD *)(v37 + 36) >= *(_DWORD *)(v38 + 36)) )
          {
            goto LABEL_59;
          }
        }
        v12 = *(_QWORD *)(v37 + 40);
LABEL_85:
        if ( !v12 )
          goto LABEL_96;
LABEL_86:
        if ( !v11 )
        {
          v14 = v10[20];
          if ( v10[14] )
            goto LABEL_21;
          goto LABEL_88;
        }
LABEL_19:
        v13 = v10[8];
        if ( v13 )
          goto LABEL_51;
        v14 = v10[20];
        if ( v10[14] )
        {
LABEL_21:
          v15 = v10[12];
          if ( v14 )
          {
            v16 = v10[18];
            v17 = *(_DWORD *)(v16 + 32);
            if ( *(_DWORD *)(v15 + 32) >= v17
              && (*(_DWORD *)(v15 + 32) != v17 || *(_DWORD *)(v15 + 36) >= *(_DWORD *)(v16 + 36)) )
            {
              goto LABEL_24;
            }
          }
          v13 = *(_QWORD *)(v15 + 40);
LABEL_50:
          if ( v13 )
          {
LABEL_51:
            v34 = v13 < v12;
            if ( v13 != v12 )
              goto LABEL_52;
            goto LABEL_90;
          }
          goto LABEL_89;
        }
LABEL_88:
        if ( v14 )
        {
          v16 = v10[18];
LABEL_24:
          v75 = (_QWORD *)(v16 + 48);
          if ( *(_QWORD *)(v16 + 64) == v16 + 48 )
            goto LABEL_89;
          v71 = j;
          v13 = 0;
          v69 = v10;
          v67 = v12;
          v18 = *(_QWORD **)(v16 + 64);
          while ( 2 )
          {
            if ( v11 )
            {
              v19 = v18[14];
              if ( v19 )
              {
LABEL_48:
                v13 += v19;
                v18 = (_QWORD *)sub_220EF30((__int64)v18);
                if ( v75 == v18 )
                {
                  j = v71;
                  v10 = v69;
                  v12 = v67;
                  goto LABEL_50;
                }
                continue;
              }
            }
            break;
          }
          v20 = v18[26];
          if ( v18[20] )
          {
            v21 = v18[18];
            if ( !v20
              || (v22 = v18[24], v23 = *(_DWORD *)(v22 + 32), *(_DWORD *)(v21 + 32) < v23)
              || *(_DWORD *)(v21 + 32) == v23 && *(_DWORD *)(v21 + 36) < *(_DWORD *)(v22 + 36) )
            {
              v19 = *(_QWORD *)(v21 + 40);
LABEL_47:
              if ( v19 )
                goto LABEL_48;
LABEL_100:
              v19 = v18[13] != 0;
              goto LABEL_48;
            }
          }
          else
          {
            if ( !v20 )
              goto LABEL_100;
            v22 = v18[24];
          }
          v24 = *(_QWORD **)(v22 + 64);
          v73 = (_QWORD *)(v22 + 48);
          if ( v24 == (_QWORD *)(v22 + 48) )
            goto LABEL_100;
          v63 = v13;
          v19 = 0;
          v65 = v18;
          v25 = v24;
          while ( 2 )
          {
            if ( v11 )
            {
              v26 = v25[14];
              if ( v26 )
                goto LABEL_45;
            }
            v27 = v25[26];
            if ( v25[20] )
            {
              v28 = v25[18];
              if ( !v27
                || (v29 = v25[24], v30 = *(_DWORD *)(v29 + 32), *(_DWORD *)(v28 + 32) < v30)
                || *(_DWORD *)(v28 + 32) == v30 && *(_DWORD *)(v28 + 36) < *(_DWORD *)(v29 + 36) )
              {
                v26 = *(_QWORD *)(v28 + 40);
                goto LABEL_44;
              }
LABEL_40:
              v31 = *(_QWORD *)(v29 + 64);
              v32 = v29 + 48;
              if ( v31 != v32 )
              {
                v61 = v25;
                v26 = 0;
                v33 = v31;
                do
                {
                  v26 += sub_EF9210((_QWORD *)(v33 + 48));
                  v33 = sub_220EF30(v33);
                }
                while ( v32 != v33 );
                v25 = v61;
LABEL_44:
                if ( v26 )
                {
LABEL_45:
                  v19 += v26;
                  v25 = (_QWORD *)sub_220EF30((__int64)v25);
                  if ( v73 == v25 )
                  {
                    v18 = v65;
                    v13 = v63;
                    goto LABEL_47;
                  }
                  continue;
                }
              }
            }
            else if ( v27 )
            {
              v29 = v25[24];
              goto LABEL_40;
            }
            break;
          }
          v26 = v25[13] != 0;
          goto LABEL_45;
        }
LABEL_89:
        v56 = v10[7] != 0;
        v34 = v56 < v12;
        if ( v56 != v12 )
        {
LABEL_52:
          v35 = v34;
          goto LABEL_53;
        }
LABEL_90:
        v57 = (int *)v79[2];
        v58 = v79[3];
        if ( v57 )
        {
          sub_C7D030(v81);
          sub_C7D280(v81, v57, v58);
          sub_C7D290(v81, v80);
          v58 = v80[0];
        }
        v59 = (int *)v10[2];
        v60 = v10[3];
        if ( v59 )
        {
          sub_C7D030(v81);
          sub_C7D280(v81, v59, v60);
          sub_C7D290(v81, v80);
          v60 = v80[0];
        }
        v35 = v60 > v58;
LABEL_53:
        j -= 8;
        if ( !v35 )
          break;
      }
      *(_QWORD *)v77 = v79;
    }
  }
}
