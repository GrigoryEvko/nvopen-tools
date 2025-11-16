// Function: sub_A57EC0
// Address: 0xa57ec0
//
_BYTE *__fastcall sub_A57EC0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  unsigned int v6; // r15d
  __int64 v7; // rax
  unsigned __int8 v8; // si
  _BYTE *result; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 *v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 *v22; // r15
  size_t v23; // rdx
  char *i; // rsi
  const char *v25; // rsi
  __int64 v26; // rdi
  __int64 *v27; // rbx
  __int64 v28; // r14
  unsigned int *v29; // rbx
  unsigned int *v30; // r14
  unsigned int v31; // r13d
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdi
  size_t v36; // rdx
  char *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  unsigned __int8 *v41; // r13
  size_t v42; // rdx
  size_t v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rdi
  int v46; // r8d
  __int64 *v47; // [rsp+8h] [rbp-38h]
  __int64 *v48; // [rsp+8h] [rbp-38h]

  v3 = *(_BYTE *)(a2 + 8);
  switch ( v3 )
  {
    case 0:
      v25 = "half";
      return (_BYTE *)sub_904010(a3, v25);
    case 1:
      v25 = "bfloat";
      return (_BYTE *)sub_904010(a3, v25);
    case 2:
      v25 = "float";
      return (_BYTE *)sub_904010(a3, v25);
    case 3:
      v25 = "double";
      return (_BYTE *)sub_904010(a3, v25);
    case 4:
      v25 = "x86_fp80";
      return (_BYTE *)sub_904010(a3, v25);
    case 5:
      v25 = "fp128";
      return (_BYTE *)sub_904010(a3, v25);
    case 6:
      v25 = "ppc_fp128";
      return (_BYTE *)sub_904010(a3, v25);
    case 7:
      v25 = "void";
      return (_BYTE *)sub_904010(a3, v25);
    case 8:
      v25 = "label";
      return (_BYTE *)sub_904010(a3, v25);
    case 9:
      v25 = "metadata";
      return (_BYTE *)sub_904010(a3, v25);
    case 10:
      v25 = "x86_amx";
      return (_BYTE *)sub_904010(a3, v25);
    case 11:
      v25 = "token";
      return (_BYTE *)sub_904010(a3, v25);
    case 12:
      v26 = sub_A51310(a3, 0x69u);
      return (_BYTE *)sub_CB59D0(v26, *(_DWORD *)(a2 + 8) >> 8);
    case 13:
      sub_A57EC0(a1, **(_QWORD **)(a2 + 16), a3);
      sub_904010(a3, " (");
      v20 = *(_QWORD *)(a2 + 16);
      v47 = (__int64 *)(v20 + 8LL * *(unsigned int *)(a2 + 12));
      if ( v47 == (__int64 *)(v20 + 8) )
      {
        if ( *(_DWORD *)(a2 + 8) >> 8 )
        {
          v36 = 0;
          v37 = 0;
LABEL_42:
          v38 = sub_A51340(a3, v37, v36);
          sub_904010(v38, "...");
        }
      }
      else
      {
        v21 = *(_QWORD *)(v20 + 8);
        v22 = (__int64 *)(v20 + 16);
        v23 = 0;
        for ( i = 0; ; i = ", " )
        {
          sub_A51340(a3, i, v23);
          sub_A57EC0(a1, v21, a3);
          if ( v47 == v22 )
            break;
          v21 = *v22;
          v23 = 2;
          ++v22;
        }
        v36 = 2;
        v37 = ", ";
        if ( *(_DWORD *)(a2 + 8) >> 8 )
          goto LABEL_42;
      }
      v8 = 41;
      return (_BYTE *)sub_A51310(a3, v8);
    case 14:
      result = (_BYTE *)sub_904010(a3, "ptr");
      v19 = *(_DWORD *)(a2 + 8) >> 8;
      if ( v19 )
      {
        v39 = sub_904010(a3, " addrspace(");
        v40 = sub_CB59D0(v39, v19);
        return (_BYTE *)sub_A51310(v40, 0x29u);
      }
      return result;
    case 15:
      if ( (*(_BYTE *)(a2 + 9) & 4) != 0 )
        return (_BYTE *)sub_A58460(a1, a2, a3);
      sub_BCB490(a2);
      if ( v12 )
      {
        v41 = (unsigned __int8 *)sub_BCB490(a2);
        v43 = v42;
        sub_A51310(a3, 0x25u);
        return sub_A54F00(a3, v41, v43);
      }
      sub_A57B80(a1);
      v13 = *(unsigned int *)(a1 + 192);
      v14 = *(_QWORD *)(a1 + 176);
      if ( !(_DWORD)v13 )
        goto LABEL_49;
      v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( a2 == *v16 )
        goto LABEL_11;
      v46 = 1;
      while ( 2 )
      {
        if ( v17 != -4096 )
        {
          v15 = (v13 - 1) & (v46 + v15);
          v16 = (__int64 *)(v14 + 16LL * v15);
          v17 = *v16;
          if ( a2 != *v16 )
          {
            ++v46;
            continue;
          }
LABEL_11:
          if ( v16 != (__int64 *)(v14 + 16 * v13) )
          {
            v18 = sub_A51310(a3, 0x25u);
            return (_BYTE *)sub_CB59D0(v18, *((unsigned int *)v16 + 2));
          }
        }
        break;
      }
LABEL_49:
      v44 = sub_904010(a3, "%\"type ");
      v45 = sub_CB5A80(v44, a2);
      return (_BYTE *)sub_A51310(v45, 0x22u);
    case 16:
      v10 = sub_A51310(a3, 0x5Bu);
      v11 = sub_CB59D0(v10, *(_QWORD *)(a2 + 32));
      sub_904010(v11, " x ");
      sub_A57EC0(a1, *(_QWORD *)(a2 + 24), a3);
      v8 = 93;
      return (_BYTE *)sub_A51310(a3, v8);
    case 17:
    case 18:
      v6 = *(_DWORD *)(a2 + 32);
      sub_904010(a3, "<");
      if ( v3 == 18 )
        sub_904010(a3, "vscale x ");
      v7 = sub_CB59D0(a3, v6);
      sub_904010(v7, " x ");
      sub_A57EC0(a1, *(_QWORD *)(a2 + 24), a3);
      v8 = 62;
      return (_BYTE *)sub_A51310(a3, v8);
    case 19:
      v33 = sub_904010(a3, "typedptr(");
      sub_A587F0(*(_QWORD *)(a2 + 24), v33, 0, 0);
      v34 = sub_904010(v33, ", ");
      v35 = sub_CB59D0(v34, *(_DWORD *)(a2 + 8) >> 8);
      return (_BYTE *)sub_904010(v35, ")");
    case 20:
      sub_904010(a3, "target(\"");
      sub_C92400(*(_QWORD *)(a2 + 24), *(_QWORD *)(a2 + 32), a3);
      sub_904010(a3, "\"");
      v27 = *(__int64 **)(a2 + 16);
      v48 = &v27[*(unsigned int *)(a2 + 12)];
      while ( v48 != v27 )
      {
        v28 = *v27++;
        sub_904010(a3, ", ");
        sub_A587F0(v28, a3, 0, 1);
      }
      v29 = *(unsigned int **)(a2 + 40);
      v30 = &v29[*(_DWORD *)(a2 + 8) >> 8];
      while ( v30 != v29 )
      {
        v31 = *v29++;
        v32 = sub_904010(a3, ", ");
        sub_CB59D0(v32, v31);
      }
      v25 = ")";
      return (_BYTE *)sub_904010(a3, v25);
    default:
      BUG();
  }
}
