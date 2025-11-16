// Function: sub_39E68E0
// Address: 0x39e68e0
//
__int64 __fastcall sub_39E68E0(__int64 a1, _BYTE *a2, int a3)
{
  unsigned __int64 v5; // r13
  unsigned int v6; // r13d
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // r14
  char *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi
  __int64 v14; // rdi
  _BYTE *v15; // rax
  char v16; // si
  char v17; // al
  char *v18; // rdx
  __int64 v19; // rdi

  switch ( a3 )
  {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      v6 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 280) + 305LL);
      if ( (_BYTE)v6 )
      {
        sub_1263B40(*(_QWORD *)(a1 + 272), "\t.type\t");
        sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
        v14 = *(_QWORD *)(a1 + 272);
        v15 = *(_BYTE **)(v14 + 24);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
        {
          v14 = sub_16E7DE0(v14, 44);
        }
        else
        {
          *(_QWORD *)(v14 + 24) = v15 + 1;
          *v15 = 44;
        }
        v16 = 64;
        v17 = 64;
        if ( **(_BYTE **)(*(_QWORD *)(a1 + 280) + 48LL) == 64 )
        {
          v17 = 37;
          v16 = 37;
        }
        v18 = *(char **)(v14 + 24);
        if ( (unsigned __int64)v18 >= *(_QWORD *)(v14 + 16) )
        {
          sub_16E7DE0(v14, v16);
        }
        else
        {
          *(_QWORD *)(v14 + 24) = v18 + 1;
          *v18 = v17;
        }
        v19 = *(_QWORD *)(a1 + 272);
        switch ( a3 )
        {
          case 2:
            sub_1263B40(v19, "gnu_indirect_function");
            break;
          case 3:
            sub_1263B40(v19, "object");
            break;
          case 4:
            sub_1263B40(v19, "tls_object");
            break;
          case 5:
            sub_1263B40(v19, "common");
            break;
          case 6:
            sub_1263B40(v19, "notype");
            break;
          case 7:
            sub_1263B40(v19, "gnu_unique_object");
            break;
          default:
            sub_1263B40(v19, "function");
            break;
        }
        sub_39E06C0(a1);
      }
      return v6;
    case 8:
      sub_1263B40(*(_QWORD *)(a1 + 272), *(const char **)(*(_QWORD *)(a1 + 280) + 288LL));
      goto LABEL_3;
    case 9:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.hidden\t");
      goto LABEL_3;
    case 10:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.indirect_symbol\t");
      goto LABEL_3;
    case 11:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.internal\t");
      goto LABEL_3;
    case 12:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.lazy_reference\t");
      goto LABEL_3;
    case 13:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.local\t");
      goto LABEL_3;
    case 14:
      v6 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 280) + 308LL);
      if ( !(_BYTE)v6 )
        return v6;
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.no_dead_strip\t");
LABEL_3:
      sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
      v5 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v10 = *(_QWORD *)(a1 + 272);
        v11 = *(char **)(a1 + 304);
        v12 = *(unsigned int *)(a1 + 312);
        v13 = *(void **)(v10 + 24);
        if ( v5 > *(_QWORD *)(v10 + 16) - (_QWORD)v13 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v11, v12);
        }
        else
        {
          memcpy(v13, v11, v12);
          *(_QWORD *)(v10 + 24) += v5;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
      {
        v6 = 1;
        sub_39E0440(a1);
      }
      else
      {
        v8 = *(_QWORD *)(a1 + 272);
        v9 = *(_BYTE **)(v8 + 24);
        v6 = 1;
        if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
        {
          sub_16E7DE0(v8, 10);
        }
        else
        {
          *(_QWORD *)(v8 + 24) = v9 + 1;
          *v9 = 10;
        }
      }
      return v6;
    case 15:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.symbol_resolver\t");
      goto LABEL_3;
    case 16:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.alt_entry\t");
      goto LABEL_3;
    case 17:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.private_extern\t");
      goto LABEL_3;
    case 18:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.protected\t");
      goto LABEL_3;
    case 19:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.reference\t");
      goto LABEL_3;
    case 20:
      sub_1263B40(*(_QWORD *)(a1 + 272), *(const char **)(*(_QWORD *)(a1 + 280) + 312LL));
      goto LABEL_3;
    case 21:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.weak_definition\t");
      goto LABEL_3;
    case 22:
      sub_1263B40(*(_QWORD *)(a1 + 272), *(const char **)(*(_QWORD *)(a1 + 280) + 320LL));
      goto LABEL_3;
    case 23:
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.weak_def_can_be_hidden\t");
      goto LABEL_3;
    default:
      goto LABEL_3;
  }
}
