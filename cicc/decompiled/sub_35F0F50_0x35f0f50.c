// Function: sub_35F0F50
// Address: 0x35f0f50
//
void __fastcall sub_35F0F50(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  _BYTE *v6; // rcx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  _QWORD *v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  size_t v20; // rdx
  char *v21; // rsi
  unsigned __int8 *v22[2]; // [rsp-38h] [rbp-38h] BYREF
  _BYTE v23[40]; // [rsp-28h] [rbp-28h] BYREF

  if ( a5 && !strcmp(a5, "name") )
  {
    switch ( (unsigned int)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) )
    {
      case '&':
        v8 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v8) > 5 )
        {
          *(_DWORD *)v8 = 1684632613;
          *(_WORD *)(v8 + 4) = 30766;
          *(_QWORD *)(a4 + 32) += 6LL;
          return;
        }
        v20 = 6;
        v21 = "%tid.x";
        goto LABEL_36;
      case '\'':
        v9 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v9) > 5 )
        {
          *(_DWORD *)v9 = 1684632613;
          *(_WORD *)(v9 + 4) = 31022;
          *(_QWORD *)(a4 + 32) += 6LL;
          return;
        }
        v20 = 6;
        v21 = "%tid.y";
        goto LABEL_36;
      case '(':
        v10 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) > 5 )
        {
          *(_DWORD *)v10 = 1684632613;
          *(_WORD *)(v10 + 4) = 31278;
          *(_QWORD *)(a4 + 32) += 6LL;
          return;
        }
        v20 = 6;
        v21 = "%tid.z";
        goto LABEL_36;
      case ')':
        v12 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v12) > 6 )
        {
          *(_DWORD *)v12 = 1769238053;
          *(_WORD *)(v12 + 4) = 11876;
          *(_BYTE *)(v12 + 6) = 120;
          *(_QWORD *)(a4 + 32) += 7LL;
          return;
        }
        v20 = 7;
        v21 = "%ntid.x";
        goto LABEL_36;
      case '*':
        v11 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 6 )
        {
          *(_DWORD *)v11 = 1769238053;
          *(_WORD *)(v11 + 4) = 11876;
          *(_BYTE *)(v11 + 6) = 121;
          *(_QWORD *)(a4 + 32) += 7LL;
          return;
        }
        v20 = 7;
        v21 = "%ntid.y";
        goto LABEL_36;
      case '+':
        v13 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v13) > 6 )
        {
          *(_DWORD *)v13 = 1769238053;
          *(_WORD *)(v13 + 4) = 11876;
          *(_BYTE *)(v13 + 6) = 122;
          *(_QWORD *)(a4 + 32) += 7LL;
          return;
        }
        v20 = 7;
        v21 = "%ntid.z";
        goto LABEL_36;
      case ',':
        v14 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v14 > 7u )
        {
          *v14 = 0x782E646961746325LL;
          *(_QWORD *)(a4 + 32) += 8LL;
          return;
        }
        v20 = 8;
        v21 = "%ctaid.x";
        goto LABEL_36;
      case '-':
        v17 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 > 7u )
        {
          *v17 = 0x792E646961746325LL;
          *(_QWORD *)(a4 + 32) += 8LL;
          return;
        }
        v20 = 8;
        v21 = "%ctaid.y";
        goto LABEL_36;
      case '.':
        v16 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v16 > 7u )
        {
          *v16 = 0x7A2E646961746325LL;
          *(_QWORD *)(a4 + 32) += 8LL;
          return;
        }
        v20 = 8;
        v21 = "%ctaid.z";
        goto LABEL_36;
      case '/':
        v18 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v18) > 8 )
        {
          *(_BYTE *)(v18 + 8) = 120;
          *(_QWORD *)v18 = 0x2E64696174636E25LL;
          *(_QWORD *)(a4 + 32) += 9LL;
          return;
        }
        v20 = 9;
        v21 = "%nctaid.x";
        goto LABEL_36;
      case '0':
        v15 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v15) > 8 )
        {
          *(_BYTE *)(v15 + 8) = 121;
          *(_QWORD *)v15 = 0x2E64696174636E25LL;
          *(_QWORD *)(a4 + 32) += 9LL;
          return;
        }
        v20 = 9;
        v21 = "%nctaid.y";
        goto LABEL_36;
      case '1':
        v19 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v19) <= 8 )
        {
          v20 = 9;
          v21 = "%nctaid.z";
LABEL_36:
          sub_CB6200(a4, (unsigned __int8 *)v21, v20);
        }
        else
        {
          *(_BYTE *)(v19 + 8) = 122;
          *(_QWORD *)v19 = 0x2E64696174636E25LL;
          *(_QWORD *)(a4 + 32) += 9LL;
        }
        break;
      case '^':
        v6 = &unk_44F0A61;
        v22[0] = v23;
        v7 = (char *)&unk_44F0A61 - 9;
        goto LABEL_5;
      case '_':
        v6 = &unk_44F0A50;
        v22[0] = v23;
        v7 = (char *)&unk_44F0A50 - 16;
LABEL_5:
        sub_35ED3A0((__int64 *)v22, (char)v7, v7, v6);
        sub_CB6200(a4, v22[0], (size_t)v22[1]);
        if ( v22[0] != v23 )
          j_j___libc_free_0((unsigned __int64)v22[0]);
        return;
      default:
        sub_C64ED0("Unhandled special register", 1u);
    }
  }
}
