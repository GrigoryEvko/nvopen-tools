// Function: sub_E7FAD0
// Address: 0xe7fad0
//
unsigned __int64 __fastcall sub_E7FAD0(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rbp
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r13
  __int64 v11; // r12
  unsigned int *v12; // rbx
  unsigned int v13; // eax
  _BYTE *v14; // rax
  __int64 v15; // r14
  unsigned __int64 result; // rax
  unsigned __int16 v17; // bx
  _BYTE *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  size_t v21; // rdx
  _BYTE *v22; // rdi
  unsigned __int8 *v23; // rsi
  unsigned __int64 v24; // rax
  size_t v25; // r13
  __int64 v26; // r14
  unsigned int v27; // eax
  __int64 v28; // rdx
  const char *v29; // rax
  _BYTE *v30; // rdi
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  _BYTE *v36; // rax
  signed __int64 v37; // rsi
  _BYTE *v38; // rax
  __int64 v39; // rax
  size_t v40; // rdx
  void *v41; // rdi
  unsigned __int8 *v42; // rsi
  _BYTE *v43; // rax
  unsigned int v44; // ebx
  void *v45; // rdx
  char v46; // si
  __int64 v47; // rdi
  _WORD *v48; // rdx
  _BYTE *v49; // rax
  _WORD *v50; // rdx
  _BYTE *v51; // rax
  _WORD *v52; // rdx
  _BYTE *v53; // rax
  _BYTE *v54; // rax
  __int64 v55; // rax
  _BYTE *v56; // rax
  _BYTE *v57; // rax
  _BYTE *v58; // rax
  _WORD *v59; // rdx
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  _WORD *v62; // rdx
  _WORD *v63; // rdx
  _BYTE *v64; // rax
  _WORD *v65; // rdx
  _WORD *v66; // rdx
  __int64 v67; // rax
  _BYTE *v68; // rax
  _BYTE *v69; // rax
  _WORD *v70; // rdx
  __int64 v71; // rax
  size_t v72; // [rsp-70h] [rbp-70h]
  __int64 v73; // [rsp-60h] [rbp-60h] BYREF
  _QWORD v74[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v75; // [rsp-48h] [rbp-48h]
  __int16 v76; // [rsp-38h] [rbp-38h]
  __int64 v77; // [rsp-28h] [rbp-28h]
  __int64 v78; // [rsp-20h] [rbp-20h]
  __int64 v79; // [rsp-18h] [rbp-18h]
  __int64 v80; // [rsp-8h] [rbp-8h]

  while ( 2 )
  {
    v80 = v7;
    v79 = v9;
    v10 = a3;
    v78 = v8;
    v11 = a2;
    v77 = v6;
    v12 = a1;
    switch ( *(_BYTE *)a1 )
    {
      case 0:
        v30 = (_BYTE *)*((_QWORD *)a1 + 2);
        if ( (unsigned __int8)(*v30 - 1) > 1u )
        {
          v32 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v32 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 40);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v32 + 1;
            *v32 = 40;
          }
          sub_E7FAD0(*((_QWORD *)v12 + 2), a2, v10, 0);
          v33 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v33 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 41);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v33 + 1;
            *v33 = 41;
          }
        }
        else
        {
          sub_E7FAD0(v30, a2, a3, 0);
        }
        if ( *v12 <= 0x13FF )
        {
          switch ( *v12 >> 8 )
          {
            case 0u:
              v55 = *((_QWORD *)v12 + 3);
              if ( *(_BYTE *)v55 == 1 )
              {
                v37 = *(_QWORD *)(v55 + 16);
                if ( v37 < 0 )
                  return sub_CB59F0(v11, v37);
              }
              v56 = *(_BYTE **)(v11 + 32);
              if ( (unsigned __int64)v56 >= *(_QWORD *)(v11 + 24) )
              {
                sub_CB5D20(v11, 43);
              }
              else
              {
                *(_QWORD *)(v11 + 32) = v56 + 1;
                *v56 = 43;
              }
              break;
            case 1u:
              v64 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v64 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 38);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v64 + 1;
                *v64 = 38;
              }
              break;
            case 2u:
              v58 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v58 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 47);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v58 + 1;
                *v58 = 47;
              }
              break;
            case 3u:
              v62 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v62 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"==", 2u);
              }
              else
              {
                *v62 = 15677;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 4u:
              v53 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v53 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 62);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v53 + 1;
                *v53 = 62;
              }
              break;
            case 5u:
              v66 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v66 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)">=", 2u);
              }
              else
              {
                *v66 = 15678;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 6u:
              v59 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v59 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"&&", 2u);
              }
              else
              {
                a6 = 9766;
                *v59 = 9766;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 7u:
              v63 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v63 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"||", 2u);
              }
              else
              {
                a5 = 31868;
                *v63 = 31868;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 8u:
              v54 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v54 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 60);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v54 + 1;
                *v54 = 60;
              }
              break;
            case 9u:
              v65 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v65 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"<=", 2u);
              }
              else
              {
                *v65 = 15676;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 0xAu:
              v57 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v57 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 37);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v57 + 1;
                *v57 = 37;
              }
              break;
            case 0xBu:
              v61 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v61 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 42);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v61 + 1;
                *v61 = 42;
              }
              break;
            case 0xCu:
              v52 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v52 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"!=", 2u);
              }
              else
              {
                *v52 = 15649;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 0xDu:
              v51 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v51 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 124);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v51 + 1;
                *v51 = 124;
              }
              break;
            case 0xEu:
              v35 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v35 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 33);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v35 + 1;
                *v35 = 33;
              }
              break;
            case 0xFu:
              v50 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v50 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)"<<", 2u);
              }
              else
              {
                *v50 = 15420;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 0x10u:
            case 0x11u:
              v48 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v48 <= 1u )
              {
                sub_CB6200(a2, (unsigned __int8 *)">>", 2u);
              }
              else
              {
                *v48 = 15934;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              break;
            case 0x12u:
              v49 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v49 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 45);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v49 + 1;
                *v49 = 45;
              }
              break;
            case 0x13u:
              v60 = *(_BYTE **)(a2 + 32);
              if ( (unsigned __int64)v60 >= *(_QWORD *)(a2 + 24) )
              {
                sub_CB5D20(a2, 94);
              }
              else
              {
                *(_QWORD *)(a2 + 32) = v60 + 1;
                *v60 = 94;
              }
              break;
          }
        }
        a1 = (unsigned int *)*((_QWORD *)v12 + 3);
        if ( (unsigned __int8)(*(_BYTE *)a1 - 1) <= 1u )
          goto LABEL_8;
        v36 = *(_BYTE **)(v11 + 32);
        if ( (unsigned __int64)v36 >= *(_QWORD *)(v11 + 24) )
        {
          sub_CB5D20(v11, 40);
        }
        else
        {
          *(_QWORD *)(v11 + 32) = v36 + 1;
          *v36 = 40;
        }
        sub_E7FAD0(*((_QWORD *)v12 + 3), v11, v10, 0);
        result = *(_QWORD *)(v11 + 32);
        if ( result >= *(_QWORD *)(v11 + 24) )
          goto LABEL_114;
        *(_QWORD *)(v11 + 32) = result + 1;
        *(_BYTE *)result = 41;
        return result;
      case 1:
        v26 = *((_QWORD *)a1 + 2);
        v27 = *a1 >> 8;
        v28 = v27 & 0x100;
        if ( (!v10 || v26 >= 0 || *(_BYTE *)(v10 + 256)) && (v27 & 0x100) == 0 )
        {
          v37 = *((_QWORD *)a1 + 2);
          return sub_CB59F0(v11, v37);
        }
        if ( (unsigned __int8)v27 == 4 )
        {
          v29 = "0x%08lx";
          goto LABEL_30;
        }
        if ( (unsigned __int8)v27 > 4u )
        {
          if ( (unsigned __int8)v27 == 8 )
          {
            v29 = "0x%016lx";
            goto LABEL_30;
          }
        }
        else
        {
          if ( (unsigned __int8)v27 == 1 )
          {
            v29 = "0x%02lx";
            goto LABEL_30;
          }
          if ( (unsigned __int8)v27 == 2 )
          {
            v29 = "0x%04lx";
LABEL_30:
            v74[1] = v29;
            v75 = v26;
            v74[0] = &unk_49DBEF0;
            return sub_CB6620(a2, (__int64)v74, v28, a4, a5, a6);
          }
        }
        v70 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v70 <= 1u )
        {
          v11 = sub_CB6200(a2, (unsigned __int8 *)"0x", 2u);
        }
        else
        {
          *v70 = 30768;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        v73 = v26;
        v74[0] = &v73;
        v75 = 0;
        v76 = 271;
        return sub_CA0E80((__int64)v74, v11);
      case 2:
        v15 = *((_QWORD *)a1 + 2);
        if ( a3 )
        {
          if ( *(_BYTE *)(a3 + 353) == 1
            && !(_BYTE)a4
            && (*(_BYTE *)(v15 + 8) & 1) != 0
            && (v67 = *(_QWORD *)(v15 - 8), *(_QWORD *)v67)
            && *(_BYTE *)(v67 + 24) == 36 )
          {
            v68 = *(_BYTE **)(a2 + 32);
            if ( (unsigned __int64)v68 >= *(_QWORD *)(a2 + 24) )
            {
              sub_CB5D20(a2, 40);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = v68 + 1;
              *v68 = 40;
            }
            sub_EA12C0(v15, a2, v10);
            v69 = *(_BYTE **)(a2 + 32);
            if ( (unsigned __int64)v69 >= *(_QWORD *)(a2 + 24) )
            {
              sub_CB5D20(a2, 41);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = v69 + 1;
              *v69 = 41;
            }
          }
          else
          {
            sub_EA12C0(*((_QWORD *)a1 + 2), a2, a3);
          }
          result = *a1 >> 8;
          v17 = result;
          if ( !(_WORD)result )
            return result;
          v18 = *(_BYTE **)(a2 + 32);
          v19 = *(_QWORD *)(a2 + 24);
          if ( !*(_BYTE *)(v10 + 352) )
          {
            if ( v19 <= (unsigned __int64)v18 )
            {
              v11 = sub_CB5D20(a2, 64);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = v18 + 1;
              *v18 = 64;
            }
            v39 = sub_106EF10(v10, v17);
            v41 = *(void **)(v11 + 32);
            v42 = (unsigned __int8 *)v39;
            result = *(_QWORD *)(v11 + 24) - (_QWORD)v41;
            if ( result < v40 )
              return sub_CB6200(v11, v42, v40);
            if ( v40 )
            {
              v72 = v40;
              result = (unsigned __int64)memcpy(v41, v42, v40);
              *(_QWORD *)(v11 + 32) += v72;
            }
            return result;
          }
          if ( v19 <= (unsigned __int64)v18 )
          {
            v11 = sub_CB5D20(a2, 40);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v18 + 1;
            *v18 = 40;
          }
          v20 = sub_106EF10(v10, v17);
          v22 = *(_BYTE **)(v11 + 32);
          v23 = (unsigned __int8 *)v20;
          v24 = *(_QWORD *)(v11 + 24);
          v25 = v21;
          if ( v24 - (unsigned __int64)v22 < v21 )
          {
            v71 = sub_CB6200(v11, v23, v21);
            v22 = *(_BYTE **)(v71 + 32);
            v11 = v71;
            v24 = *(_QWORD *)(v71 + 24);
          }
          else if ( v21 )
          {
            memcpy(v22, v23, v21);
            v24 = *(_QWORD *)(v11 + 24);
            v22 = (_BYTE *)(v25 + *(_QWORD *)(v11 + 32));
            *(_QWORD *)(v11 + 32) = v22;
          }
          if ( v24 > (unsigned __int64)v22 )
          {
            *(_QWORD *)(v11 + 32) = v22 + 1;
            *v22 = 41;
            return (unsigned __int64)(v22 + 1);
          }
LABEL_114:
          v46 = 41;
          v47 = v11;
        }
        else
        {
          sub_EA12C0(*((_QWORD *)a1 + 2), a2, 0);
          result = *a1;
          v44 = *a1 >> 8;
          if ( (result & 0xFFFF00) == 0 )
            return result;
          v45 = *(void **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v45 <= 9u )
          {
            v11 = sub_CB6200(a2, "@<variant ", 0xAu);
          }
          else
          {
            qmemcpy(v45, "@<variant ", 10);
            *(_QWORD *)(a2 + 32) += 10LL;
          }
          v46 = 62;
          v47 = sub_CB59F0(v11, (unsigned __int16)v44);
          result = *(_QWORD *)(v47 + 32);
          if ( result < *(_QWORD *)(v47 + 24) )
          {
            *(_QWORD *)(v47 + 32) = result + 1;
            *(_BYTE *)result = 62;
            return result;
          }
        }
        return sub_CB5D20(v47, v46);
      case 3:
        v13 = *a1 >> 8;
        if ( v13 == 2 )
        {
          v43 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v43 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 126);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v43 + 1;
            *v43 = 126;
          }
        }
        else if ( v13 > 2 )
        {
          if ( v13 == 3 )
          {
            v31 = *(_BYTE **)(a2 + 32);
            if ( (unsigned __int64)v31 >= *(_QWORD *)(a2 + 24) )
            {
              sub_CB5D20(a2, 43);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = v31 + 1;
              *v31 = 43;
            }
          }
        }
        else if ( v13 )
        {
          v14 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v14 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 45);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v14 + 1;
            *v14 = 45;
          }
        }
        else
        {
          v38 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v38 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 33);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v38 + 1;
            *v38 = 33;
          }
        }
        a1 = (unsigned int *)*((_QWORD *)a1 + 2);
        if ( *(_BYTE *)a1 )
        {
LABEL_8:
          a3 = v10;
          a2 = v11;
          a4 = 0;
          v6 = v77;
          v8 = v78;
          v9 = v79;
          v7 = v80;
          continue;
        }
        v34 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v34 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"(", 1u);
        }
        else
        {
          *v34 = 40;
          ++*(_QWORD *)(a2 + 32);
        }
        sub_E7FAD0(*((_QWORD *)v12 + 2), a2, v10, 0);
        result = *(_QWORD *)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) != result )
        {
          *(_BYTE *)result = 41;
          ++*(_QWORD *)(a2 + 32);
          return result;
        }
        v40 = 1;
        v42 = (unsigned __int8 *)")";
        return sub_CB6200(v11, v42, v40);
      case 4:
        return (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64))(*((_QWORD *)a1 - 1) + 24LL))(
                 a1 - 2,
                 a2,
                 a3);
      default:
        BUG();
    }
  }
}
