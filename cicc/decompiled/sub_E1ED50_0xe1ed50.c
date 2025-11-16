// Function: sub_E1ED50
// Address: 0xe1ed50
//
__int64 __fastcall sub_E1ED50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rsi
  int v10; // ebx
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  signed __int64 v16; // r13
  char *v17; // rax
  char v18; // cl
  _BYTE *v19; // rbx
  __int64 v20; // r8
  _BYTE *v21; // rcx
  __int64 v22; // rsi
  char *v23; // rdx
  _BYTE *v24; // rbx
  __int64 v25; // r8
  _BYTE *v26; // rcx
  char v27; // dl
  _BYTE *v28; // rbx
  __int64 v29; // r8
  _BYTE *v30; // rcx
  char v31; // dl
  char *v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  char *v37; // rax
  char *v38; // rcx
  char v39; // dl
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rbx
  __int64 v43; // r8
  __int64 v44; // r9
  char *v45; // rax
  char v46; // dl
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rbx
  __int64 v50; // r8
  __int64 v51; // r9
  char *v52; // rax
  char v53; // dl
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  char v58; // dl
  __int64 v59; // [rsp-30h] [rbp-30h]

  v6 = *(char **)a1;
  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 == *(_QWORD *)a1 || *v6 != 76 )
    return 0;
  v9 = (__int64)(v6 + 1);
  *(_QWORD *)a1 = v6 + 1;
  if ( v6 + 1 != (char *)v7 )
  {
    v10 = (unsigned __int8)v6[1];
    switch ( (char)v10 )
    {
      case 'A':
        v42 = sub_E1AEA0(a1, v9, v7, (unsigned __int8)(v10 - 65), a5);
        if ( !v42 )
          return 0;
        v45 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v45 != 69 )
          return 0;
        *(_QWORD *)a1 = v45 + 1;
        result = sub_E0E790(a1 + 816, 24, v40, v41, v43, v44);
        if ( result )
        {
          v46 = *(_BYTE *)(result + 10);
          *(_WORD *)(result + 8) = 16458;
          *(_QWORD *)(result + 16) = v42;
          *(_BYTE *)(result + 10) = v46 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49E0A48;
        }
        return result;
      case 'D':
        if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "Dn") )
          return 0;
        v37 = *(char **)a1;
        v38 = *(char **)(a1 + 8);
        if ( *(char **)a1 == v38 )
          return 0;
        v39 = *v37;
        if ( *v37 != 48 )
          goto LABEL_61;
        *(_QWORD *)a1 = v37 + 1;
        if ( v38 == v37 + 1 )
          return 0;
        v39 = *++v37;
LABEL_61:
        if ( v39 != 69 )
          return 0;
        *(_QWORD *)a1 = v37 + 1;
        return sub_E0FD70(a1 + 816, "nullptr");
      case 'T':
        return 0;
      case 'U':
        if ( v7 - v9 == 1 )
          return 0;
        if ( v6[2] != 108 )
          return 0;
        v49 = sub_E1E1C0((char **)a1, 0);
        if ( !v49 )
          return 0;
        v52 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v52 != 69 )
          return 0;
        *(_QWORD *)a1 = v52 + 1;
        result = sub_E0E790(a1 + 816, 24, v47, v48, v50, v51);
        if ( result )
        {
          v53 = *(_BYTE *)(result + 10);
          *(_WORD *)(result + 8) = 16459;
          *(_QWORD *)(result + 16) = v49;
          *(_BYTE *)(result + 10) = v53 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49E0AA8;
        }
        return result;
      case '_':
        if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 2u, &unk_3C1BC40) )
          return 0;
        result = sub_E1C560((const void **)a1, 1);
        if ( !result )
          return 0;
        v32 = *(char **)a1;
        if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v32 != 69 )
          return 0;
        *(_QWORD *)a1 = v32 + 1;
        return result;
      case 'a':
        v22 = 11;
        v23 = "signed char";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'b':
        if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 3u, "b0E") )
        {
          result = sub_E0E790(a1 + 816, 16, v33, v34, v35, v36);
          if ( result )
          {
            *(_QWORD *)result = &unk_49E09E8;
            *(_DWORD *)(result + 8) = *(_DWORD *)(result + 8) & 0xF00000 | 0x54049;
          }
        }
        else
        {
          if ( !(unsigned __int8)sub_E0F5E0((const void **)a1, 3u, "b1E") )
            return 0;
          result = sub_E0E790(a1 + 816, 16, v54, v55, v56, v57);
          if ( result )
          {
            *(_QWORD *)result = &unk_49E09E8;
            *(_DWORD *)(result + 8) = *(_DWORD *)(result + 8) & 0xF00000 | 0x1054049;
          }
        }
        return result;
      case 'c':
        v22 = 4;
        v23 = "char";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'd':
        v28 = v6 + 2;
        v29 = (__int64)(v6 + 18);
        *(_QWORD *)a1 = v6 + 2;
        v30 = v6 + 2;
        if ( (unsigned __int64)(v7 - (_QWORD)(v6 + 2)) <= 0x10 )
          return 0;
        while ( (unsigned __int8)(*v30 - 48) <= 9u || (unsigned __int8)(*v30 - 97) <= 5u )
        {
          if ( (_BYTE *)v29 == ++v30 )
          {
            *(_QWORD *)a1 = v29;
            if ( v7 == v29 || v6[18] != 69 )
              return 0;
            *(_QWORD *)a1 = v6 + 19;
            result = sub_E0E790(a1 + 816, 32, v7, (__int64)v30, v29, a6);
            if ( result )
            {
              v31 = *(_BYTE *)(result + 10);
              *(_WORD *)(result + 8) = 16463;
              *(_QWORD *)(result + 16) = 16;
              *(_QWORD *)(result + 24) = v28;
              *(_BYTE *)(result + 10) = v31 & 0xF0 | 5;
              *(_QWORD *)result = &unk_49E0DA8;
            }
            return result;
          }
        }
        return 0;
      case 'e':
        v19 = v6 + 2;
        v20 = (__int64)(v6 + 22);
        *(_QWORD *)a1 = v6 + 2;
        v21 = v6 + 2;
        if ( (unsigned __int64)(v7 - (_QWORD)(v6 + 2)) <= 0x14 )
          return 0;
        while ( (unsigned __int8)(*v21 - 48) <= 9u || (unsigned __int8)(*v21 - 97) <= 5u )
        {
          if ( (_BYTE *)v20 == ++v21 )
          {
            *(_QWORD *)a1 = v20;
            if ( v7 == v20 || v6[22] != 69 )
              return 0;
            *(_QWORD *)a1 = v6 + 23;
            result = sub_E0E790(a1 + 816, 32, v7, (__int64)v21, v20, a6);
            if ( result )
            {
              v58 = *(_BYTE *)(result + 10);
              *(_WORD *)(result + 8) = 16464;
              *(_QWORD *)(result + 16) = 20;
              *(_QWORD *)(result + 24) = v19;
              *(_BYTE *)(result + 10) = v58 & 0xF0 | 5;
              *(_QWORD *)result = &unk_49E0E08;
            }
            return result;
          }
        }
        return 0;
      case 'f':
        v24 = v6 + 2;
        v25 = (__int64)(v6 + 10);
        *(_QWORD *)a1 = v6 + 2;
        v26 = v6 + 2;
        if ( (unsigned __int64)(v7 - (_QWORD)(v6 + 2)) <= 8 )
          return 0;
        break;
      case 'h':
        v22 = 13;
        v23 = "unsigned char";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'i':
        v22 = 0;
        v23 = (char *)byte_3F871B3;
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'j':
        v22 = 1;
        v23 = (char *)"u";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'l':
        v22 = 1;
        v23 = "l";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'm':
        v22 = 2;
        v23 = "ul";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'n':
        v22 = 8;
        v23 = "__int128";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'o':
        v22 = 17;
        v23 = "unsigned __int128";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 's':
        v22 = 5;
        v23 = "short";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 't':
        v22 = 14;
        v23 = "unsigned short";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'w':
        v22 = 7;
        v23 = "wchar_t";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'x':
        v22 = 2;
        v23 = "ll";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      case 'y':
        v22 = 3;
        v23 = "ull";
        *(_QWORD *)a1 = v6 + 2;
        return sub_E0EFF0(a1, v22, (__int64)v23);
      default:
        a4 = (unsigned int)(v10 - 65);
        goto LABEL_6;
    }
    while ( (unsigned __int8)(*v26 - 48) <= 9u || (unsigned __int8)(*v26 - 97) <= 5u )
    {
      if ( (_BYTE *)v25 == ++v26 )
      {
        *(_QWORD *)a1 = v25;
        if ( v7 == v25 || v6[10] != 69 )
          return 0;
        *(_QWORD *)a1 = v6 + 11;
        result = sub_E0E790(a1 + 816, 32, v7, (__int64)v26, v25, a6);
        if ( result )
        {
          v27 = *(_BYTE *)(result + 10);
          *(_WORD *)(result + 8) = 16462;
          *(_QWORD *)(result + 16) = 8;
          *(_QWORD *)(result + 24) = v24;
          *(_BYTE *)(result + 10) = v27 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49E0D48;
        }
        return result;
      }
    }
    return 0;
  }
LABEL_6:
  v11 = sub_E1AEA0(a1, v9, v7, a4, a5);
  if ( !v11 )
    return 0;
  v16 = sub_E0DEF0((char **)a1, 1);
  if ( !v16 )
    return 0;
  v17 = *(char **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v17 != 69 )
    return 0;
  v59 = v12;
  *(_QWORD *)a1 = v17 + 1;
  result = sub_E0E790(a1 + 816, 40, v12, v13, v14, v15);
  if ( result )
  {
    v18 = *(_BYTE *)(result + 10);
    *(_QWORD *)(result + 16) = v11;
    *(_WORD *)(result + 8) = 16460;
    *(_QWORD *)(result + 24) = v16;
    *(_QWORD *)(result + 32) = v59;
    *(_BYTE *)(result + 10) = v18 & 0xF0 | 5;
    *(_QWORD *)result = &unk_49E0B08;
  }
  return result;
}
