// Function: sub_21E86B0
// Address: 0x21e86b0
//
void __fastcall sub_21E86B0(__int64 a1, unsigned int a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // rax
  char *v6; // rdi
  _BYTE *v7; // rsi
  __int64 v8; // rcx
  bool v10; // cf
  bool v11; // zf
  const char *v12; // r13
  size_t v13; // rax
  void *v14; // rdi
  size_t v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // rdx
  _QWORD *v24; // rdx
  _QWORD *v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  size_t v28; // rdx
  char *v29; // rsi
  _BYTE v30[32]; // [rsp-20h] [rbp-20h] BYREF

  if ( a4 )
  {
    v5 = a2;
    v6 = "name";
    v7 = a4;
    v8 = 5;
    v10 = (unsigned __int64)v30 < 8;
    v11 = v30 == 0;
    do
    {
      if ( !v8 )
        break;
      v10 = *v7 < (unsigned __int8)*v6;
      v11 = *v7++ == (unsigned __int8)*v6++;
      --v8;
    }
    while ( v11 );
    if ( (!v10 && !v11) == v10 )
    {
      switch ( (unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16 * v5 + 8) )
      {
        case '&':
          v16 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v16) > 5 )
          {
            *(_DWORD *)v16 = 1684632613;
            *(_WORD *)(v16 + 4) = 30766;
            *(_QWORD *)(a3 + 24) += 6LL;
            return;
          }
          v28 = 6;
          v29 = "%tid.x";
          goto LABEL_41;
        case '\'':
          v17 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v17) > 5 )
          {
            *(_DWORD *)v17 = 1684632613;
            *(_WORD *)(v17 + 4) = 31022;
            *(_QWORD *)(a3 + 24) += 6LL;
            return;
          }
          v28 = 6;
          v29 = "%tid.y";
          goto LABEL_41;
        case '(':
          v18 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v18) > 5 )
          {
            *(_DWORD *)v18 = 1684632613;
            *(_WORD *)(v18 + 4) = 31278;
            *(_QWORD *)(a3 + 24) += 6LL;
            return;
          }
          v28 = 6;
          v29 = "%tid.z";
          goto LABEL_41;
        case ')':
          v20 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v20) > 6 )
          {
            *(_DWORD *)v20 = 1769238053;
            *(_WORD *)(v20 + 4) = 11876;
            *(_BYTE *)(v20 + 6) = 120;
            *(_QWORD *)(a3 + 24) += 7LL;
            return;
          }
          v28 = 7;
          v29 = "%ntid.x";
          goto LABEL_41;
        case '*':
          v19 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v19) > 6 )
          {
            *(_DWORD *)v19 = 1769238053;
            *(_WORD *)(v19 + 4) = 11876;
            *(_BYTE *)(v19 + 6) = 121;
            *(_QWORD *)(a3 + 24) += 7LL;
            return;
          }
          v28 = 7;
          v29 = "%ntid.y";
          goto LABEL_41;
        case '+':
          v21 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v21) > 6 )
          {
            *(_DWORD *)v21 = 1769238053;
            *(_WORD *)(v21 + 4) = 11876;
            *(_BYTE *)(v21 + 6) = 122;
            *(_QWORD *)(a3 + 24) += 7LL;
            return;
          }
          v28 = 7;
          v29 = "%ntid.z";
          goto LABEL_41;
        case ',':
          v22 = *(_QWORD **)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - (_QWORD)v22 > 7u )
          {
            *v22 = 0x782E646961746325LL;
            *(_QWORD *)(a3 + 24) += 8LL;
            return;
          }
          v28 = 8;
          v29 = "%ctaid.x";
          goto LABEL_41;
        case '-':
          v25 = *(_QWORD **)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - (_QWORD)v25 > 7u )
          {
            *v25 = 0x792E646961746325LL;
            *(_QWORD *)(a3 + 24) += 8LL;
            return;
          }
          v28 = 8;
          v29 = "%ctaid.y";
          goto LABEL_41;
        case '.':
          v24 = *(_QWORD **)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - (_QWORD)v24 > 7u )
          {
            *v24 = 0x7A2E646961746325LL;
            *(_QWORD *)(a3 + 24) += 8LL;
            return;
          }
          v28 = 8;
          v29 = "%ctaid.z";
          goto LABEL_41;
        case '/':
          v27 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v27) > 8 )
          {
            *(_BYTE *)(v27 + 8) = 120;
            *(_QWORD *)v27 = 0x2E64696174636E25LL;
            *(_QWORD *)(a3 + 24) += 9LL;
            return;
          }
          v28 = 9;
          v29 = "%nctaid.x";
          goto LABEL_41;
        case '0':
          v23 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v23) > 8 )
          {
            *(_BYTE *)(v23 + 8) = 121;
            *(_QWORD *)v23 = 0x2E64696174636E25LL;
            *(_QWORD *)(a3 + 24) += 9LL;
            return;
          }
          v28 = 9;
          v29 = "%nctaid.y";
          goto LABEL_41;
        case '1':
          v26 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v26) > 8 )
          {
            *(_BYTE *)(v26 + 8) = 122;
            *(_QWORD *)v26 = 0x2E64696174636E25LL;
            *(_QWORD *)(a3 + 24) += 9LL;
            return;
          }
          v28 = 9;
          v29 = "%nctaid.z";
          goto LABEL_41;
        case '^':
          v12 = (const char *)sub_3958DA0(0, v7);
          if ( v12 )
            goto LABEL_8;
          return;
        case '_':
          v12 = (const char *)sub_3958DA0(1, v7);
          if ( !v12 )
            return;
LABEL_8:
          v13 = strlen(v12);
          v14 = *(void **)(a3 + 24);
          v15 = v13;
          if ( v13 > *(_QWORD *)(a3 + 16) - (_QWORD)v14 )
          {
            v28 = v13;
            v29 = (char *)v12;
LABEL_41:
            sub_16E7EE0(a3, v29, v28);
          }
          else if ( v13 )
          {
            memcpy(v14, v12, v13);
            *(_QWORD *)(a3 + 24) += v15;
          }
          break;
        default:
          sub_16BD130("Unhandled special register", 1u);
      }
    }
  }
}
