// Function: sub_36DFE20
// Address: 0x36dfe20
//
char __fastcall sub_36DFE20(__int64 a1, __int64 a2, int a3, __m128i a4)
{
  __int64 v4; // rbx
  void *v5; // r8
  char result; // al
  __int64 v7; // rdi
  unsigned int v8; // ebx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // ebx
  unsigned __int64 v13; // rdx
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // esi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  unsigned int v23; // ebx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  unsigned __int64 v27; // rdx
  __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rbx
  __int64 v38; // rbx
  void *v39; // r8
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int8 v43; // si
  int v44; // edx
  unsigned __int8 v45; // si
  unsigned __int8 v46; // cl
  __int64 v47; // rax
  unsigned __int8 v48; // si
  int v49; // edx
  unsigned __int8 v50; // si
  unsigned __int8 v51; // cl
  __int64 v52; // rax
  unsigned __int8 v53; // si
  int v54; // edx
  unsigned __int8 v55; // si
  unsigned __int8 v56; // cl
  __int64 v57; // rax
  unsigned __int8 v58; // si
  int v59; // edx
  unsigned __int8 v60; // si
  unsigned __int8 v61; // cl
  int v62; // edx
  __int64 v63; // rbx
  void *v64; // r8
  __int64 v65; // rdi
  unsigned int v66; // edx
  __int64 *v67; // r13
  __int64 v68; // r13
  __int64 v69; // rdi
  unsigned int v70; // edx
  __int64 *v71; // r13
  __int64 v72; // r13
  __int64 v73; // r12
  unsigned int v74; // ebx
  __int64 v75; // rdi
  __int64 v76; // r14
  unsigned int v77; // edx
  __int64 v78; // r12
  unsigned int v79; // ebx
  __int64 v80; // rdi
  unsigned int v81; // edx
  int v82; // eax
  __int64 v83; // rdi
  __int64 v84; // rdi

  switch ( a3 )
  {
    case 0:
      v62 = 0;
      return sub_36DFD20(a1, a2, v62);
    case 1:
      v62 = 3;
      return sub_36DFD20(a1, a2, v62);
    case 2:
      v62 = 1;
      return sub_36DFD20(a1, a2, v62);
    case 3:
      return *(_WORD *)(a2 + 96) == 7;
    case 4:
      return *(_WORD *)(a2 + 96) == 8;
    case 5:
      v57 = *(_QWORD *)(a2 + 112);
      v58 = *(_BYTE *)(v57 + 37);
      v59 = v58 & 0xF;
      v60 = v58 >> 4;
      v61 = *(_BYTE *)(v57 + 37) & 0xF;
      result = v59 == 4 && v60 == 5;
      if ( result )
        return 0;
      if ( v59 != 5 || v60 != 4 )
      {
        if ( !byte_3F70480[8 * v61 + v60] )
          v59 = v60;
        return v59 == 4;
      }
      return result;
    case 6:
      v52 = *(_QWORD *)(a2 + 112);
      v53 = *(_BYTE *)(v52 + 37);
      v54 = v53 & 0xF;
      v55 = v53 >> 4;
      v56 = *(_BYTE *)(v52 + 37) & 0xF;
      result = v54 == 4 && v55 == 5;
      if ( result )
        return 0;
      if ( v54 != 5 || v55 != 4 )
      {
        if ( !byte_3F70480[8 * v56 + v55] )
          v54 = v55;
        return v54 == 5;
      }
      return result;
    case 7:
      v47 = *(_QWORD *)(a2 + 112);
      v48 = *(_BYTE *)(v47 + 37);
      v49 = v48 & 0xF;
      v50 = v48 >> 4;
      v51 = *(_BYTE *)(v47 + 37) & 0xF;
      result = v49 == 4 && v50 == 5;
      if ( !result )
      {
        result = v49 == 5 && v50 == 4;
        if ( !result )
        {
          if ( !byte_3F70480[8 * v51 + v50] )
            v49 = v50;
          return v49 == 6;
        }
      }
      return result;
    case 8:
      v42 = *(_QWORD *)(a2 + 112);
      v43 = *(_BYTE *)(v42 + 37);
      v44 = v43 & 0xF;
      v45 = v43 >> 4;
      v46 = *(_BYTE *)(v42 + 37) & 0xF;
      result = v44 == 4 && v45 == 5;
      if ( result )
        return 0;
      if ( v44 != 5 || v45 != 4 )
      {
        if ( !byte_3F70480[8 * v46 + v45] )
          v44 = v45;
        return v44 == 2;
      }
      return result;
    case 9:
      v41 = *(_QWORD *)(a2 + 56);
      result = 0;
      if ( v41 )
        return *(_QWORD *)(v41 + 32) == 0;
      return result;
    case 10:
      return *(_WORD *)(a2 + 96) == 6;
    case 11:
      return sub_C41C30((__int64 *)(*(_QWORD *)(a2 + 96) + 24LL), a4) == 0.0;
    case 12:
      return sub_C41B00((__int64 *)(*(_QWORD *)(a2 + 96) + 24LL)) == 0.0;
    case 13:
      v40 = *(_QWORD *)(a2 + 56);
      result = 0;
      if ( v40 )
      {
        if ( !*(_QWORD *)(v40 + 32) )
        {
          result = 1;
          if ( (*(_BYTE *)(a2 + 28) & 0x20) == 0 )
            return (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 864LL) & 4) != 0;
        }
      }
      return result;
    case 14:
      result = 1;
      if ( *(char *)(a2 + 28) >= 0 )
        return (*(_BYTE *)(*(_QWORD *)(a1 + 952) + 864LL) & 0x10) != 0;
      return result;
    case 15:
      return sub_C41C30((__int64 *)(*(_QWORD *)(a2 + 96) + 24LL), a4) == 1.0;
    case 16:
      return sub_C41B00((__int64 *)(*(_QWORD *)(a2 + 96) + 24LL)) == 1.0;
    case 17:
      v69 = *(_QWORD *)(a2 + 96);
      v70 = *(_DWORD *)(v69 + 32);
      v71 = *(__int64 **)(v69 + 24);
      if ( v70 <= 0x40 )
      {
        result = 1;
        if ( v70 )
        {
          result = 0;
          v72 = (__int64)((_QWORD)v71 << (64 - (unsigned __int8)v70)) >> (64 - (unsigned __int8)v70);
          if ( v72 >= 0 )
            return v72 <= 30;
        }
        return result;
      }
      v73 = v69 + 24;
      v74 = v70 + 1;
      v75 = v69 + 24;
      v76 = v71[(v70 - 1) >> 6] & (1LL << ((unsigned __int8)v70 - 1));
      if ( v76 )
      {
        if ( v74 - (unsigned int)sub_C44500(v75) > 0x40 || *v71 < 0 )
          return 0;
        v77 = v74 - sub_C44500(v73);
        goto LABEL_97;
      }
      if ( v74 - (unsigned int)sub_C444A0(v75) > 0x40 )
        return v76 != 0;
      result = 0;
      if ( *v71 >= 0 )
      {
        v77 = v74 - sub_C444A0(v73);
LABEL_97:
        if ( v77 > 0x40 )
          return v76 != 0;
        return *v71 <= 30;
      }
      return result;
    case 18:
      v65 = *(_QWORD *)(a2 + 96);
      v66 = *(_DWORD *)(v65 + 32);
      v67 = *(__int64 **)(v65 + 24);
      if ( v66 > 0x40 )
      {
        v78 = v65 + 24;
        v79 = v66 + 1;
        v80 = v65 + 24;
        v76 = v67[(v66 - 1) >> 6] & (1LL << ((unsigned __int8)v66 - 1));
        if ( v76 )
        {
          v81 = v79 - sub_C44500(v80);
          result = 0;
          if ( v81 > 0x40 || *v67 < 0 )
            return result;
          v82 = sub_C44500(v78);
        }
        else
        {
          if ( v79 - (unsigned int)sub_C444A0(v80) <= 0x40 && *v67 < 0 )
            return 0;
          v82 = sub_C444A0(v78);
        }
        if ( v79 - v82 > 0x40 )
          return v76 != 0;
        else
          return *v67 <= 14;
      }
      else
      {
        result = 1;
        if ( v66 )
        {
          result = 0;
          v68 = (__int64)((_QWORD)v67 << (64 - (unsigned __int8)v66)) >> (64 - (unsigned __int8)v66);
          if ( v68 >= 0 )
            return v68 <= 14;
        }
      }
      return result;
    case 19:
      v38 = *(_QWORD *)(a2 + 96);
      v39 = sub_C33310();
      result = 0;
      if ( *(void **)(v38 + 24) == v39 )
        return sub_C41C30((__int64 *)(v38 + 24), a4) == 1.0;
      return result;
    case 20:
      v36 = *(_QWORD *)(a2 + 96);
      if ( *(void **)(v36 + 24) == sub_C33340() )
        v37 = *(_QWORD *)(v36 + 32);
      else
        v37 = v36 + 24;
      return (*(_BYTE *)(v37 + 20) & 7) == 3;
    case 21:
      v34 = *(_QWORD *)(a2 + 96);
      v35 = *(_QWORD **)(v34 + 24);
      if ( *(_DWORD *)(v34 + 32) > 0x40u )
        v35 = (_QWORD *)*v35;
      return v35 == 0;
    case 22:
      v32 = *(_QWORD *)(a2 + 96);
      v33 = *(_QWORD **)(v32 + 24);
      if ( *(_DWORD *)(v32 + 32) > 0x40u )
        v33 = (_QWORD *)*v33;
      return v33 == (_QWORD *)1;
    case 23:
      v30 = *(_QWORD *)(a2 + 96);
      v31 = *(_QWORD **)(v30 + 24);
      if ( *(_DWORD *)(v30 + 32) > 0x40u )
        v31 = (_QWORD *)*v31;
      return v31 == (_QWORD *)2;
    case 24:
      v28 = *(_QWORD *)(a2 + 96);
      v29 = *(_QWORD **)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) > 0x40u )
        v29 = (_QWORD *)*v29;
      return v29 == (_QWORD *)3;
    case 25:
      v22 = *(_QWORD *)(a2 + 96);
      v23 = *(_DWORD *)(v22 + 32);
      v24 = *(_QWORD *)(v22 + 24);
      v25 = 1LL << ((unsigned __int8)v23 - 1);
      if ( v23 > 0x40 )
      {
        v84 = v22 + 24;
        if ( (*(_QWORD *)(v24 + 8LL * ((v23 - 1) >> 6)) & v25) != 0 )
          v26 = sub_C44500(v84);
        else
          v26 = sub_C444A0(v84);
      }
      else if ( (v25 & v24) != 0 )
      {
        if ( v23 )
        {
          v26 = 64;
          v27 = ~(v24 << (64 - (unsigned __int8)v23));
          if ( v27 )
          {
            _BitScanReverse64(&v27, v27);
            v26 = v27 ^ 0x3F;
          }
        }
        else
        {
          v26 = 0;
        }
      }
      else
      {
        v26 = *(_DWORD *)(v22 + 32);
        if ( v24 )
        {
          _BitScanReverse64(&v24, v24);
          v26 = v23 - 64 + (v24 ^ 0x3F);
        }
      }
      return v23 + 1 - v26 <= 0x20;
    case 26:
      v16 = *(_QWORD *)(a2 + 96);
      v17 = *(_DWORD *)(v16 + 32);
      v18 = *(_QWORD *)(v16 + 24);
      v8 = v17 + 1;
      v19 = 1LL << ((unsigned __int8)v17 - 1);
      if ( v17 > 0x40 )
      {
        v83 = v16 + 24;
        if ( (*(_QWORD *)(v18 + 8LL * ((v17 - 1) >> 6)) & v19) == 0 )
          return v8 - (unsigned int)sub_C444A0(v83) <= 0x10;
        return v8 - (unsigned int)sub_C44500(v83) <= 0x10;
      }
      else if ( (v19 & v18) != 0 )
      {
        if ( v17 )
        {
          v20 = ~(v18 << (64 - (unsigned __int8)v17));
          if ( v20 )
          {
            _BitScanReverse64(&v21, v20);
            return v8 - ((unsigned int)v21 ^ 0x3F) <= 0x10;
          }
          else
          {
            return v17 - 63 <= 0x10;
          }
        }
        else
        {
          return 1;
        }
      }
      else
      {
        result = 1;
        if ( v18 )
        {
          _BitScanReverse64(&v18, v18);
          return 65 - ((unsigned int)v18 ^ 0x3F) <= 0x10;
        }
      }
      return result;
    case 27:
      v11 = *(_QWORD *)(a2 + 96);
      v12 = *(_DWORD *)(v11 + 32);
      if ( v12 > 0x40 )
      {
        v14 = sub_C444A0(v11 + 24);
      }
      else
      {
        v13 = *(_QWORD *)(v11 + 24);
        v14 = *(_DWORD *)(v11 + 32);
        if ( v13 )
        {
          _BitScanReverse64(&v15, v13);
          v14 = v12 - 64 + (v15 ^ 0x3F);
        }
      }
      return v12 - v14 <= 0x20;
    case 28:
      v7 = *(_QWORD *)(a2 + 96);
      v8 = *(_DWORD *)(v7 + 32);
      if ( v8 > 0x40 )
      {
        v83 = v7 + 24;
        return v8 - (unsigned int)sub_C444A0(v83) <= 0x10;
      }
      else
      {
        v9 = *(_QWORD *)(v7 + 24);
        result = 1;
        if ( v9 )
        {
          _BitScanReverse64(&v10, v9);
          return 64 - ((unsigned int)v10 ^ 0x3F) <= 0x10;
        }
      }
      return result;
    case 29:
      v4 = *(_QWORD *)(a2 + 96);
      v5 = sub_C33320();
      result = 0;
      if ( *(void **)(v4 + 24) == v5 )
        return sub_C41B00((__int64 *)(v4 + 24)) == 1.0;
      return result;
    case 30:
      v63 = *(_QWORD *)(a2 + 96);
      v64 = sub_C33320();
      result = 0;
      if ( *(void **)(v63 + 24) == v64 )
        return sub_C41B00((__int64 *)(v63 + 24)) == -1.0;
      return result;
    default:
      BUG();
  }
}
