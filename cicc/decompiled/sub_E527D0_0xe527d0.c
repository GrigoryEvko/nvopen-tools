// Function: sub_E527D0
// Address: 0xe527d0
//
__int64 __fastcall sub_E527D0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v5; // r13d
  unsigned __int8 v7; // si
  __int64 v8; // rdi
  __int64 v9; // rdi

  switch ( a3 )
  {
    case 0:
      BUG();
    case 1:
    case 13:
      return 0;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
      v5 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 312) + 289LL);
      if ( (_BYTE)v5 )
      {
        sub_904010(*(_QWORD *)(a1 + 304), "\t.type\t");
        sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
        v7 = 64;
        v8 = sub_A51310(*(_QWORD *)(a1 + 304), 0x2Cu);
        if ( **(_BYTE **)(*(_QWORD *)(a1 + 312) + 48LL) == 64 )
          v7 = 37;
        sub_A51310(v8, v7);
        v9 = *(_QWORD *)(a1 + 304);
        switch ( a3 )
        {
          case 3:
            sub_904010(v9, "gnu_indirect_function");
            break;
          case 4:
            sub_904010(v9, "object");
            break;
          case 5:
            sub_904010(v9, "tls_object");
            break;
          case 6:
            sub_904010(v9, "common");
            break;
          case 7:
            sub_904010(v9, "notype");
            break;
          case 8:
            sub_904010(v9, "gnu_unique_object");
            break;
          default:
            sub_904010(*(_QWORD *)(a1 + 304), "function");
            break;
        }
        sub_E4D880(a1);
      }
      return v5;
    case 9:
      sub_904010(*(_QWORD *)(a1 + 304), *(const char **)(*(_QWORD *)(a1 + 312) + 272LL));
      goto LABEL_3;
    case 10:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.lglobl\t");
      goto LABEL_3;
    case 11:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.extern\t");
      goto LABEL_3;
    case 12:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.hidden\t");
      goto LABEL_3;
    case 14:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.indirect_symbol\t");
      goto LABEL_3;
    case 15:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.internal\t");
      goto LABEL_3;
    case 16:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.lazy_reference\t");
      goto LABEL_3;
    case 17:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.local\t");
      goto LABEL_3;
    case 18:
      v5 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 312) + 292LL);
      if ( !(_BYTE)v5 )
        return v5;
      sub_904010(*(_QWORD *)(a1 + 304), "\t.no_dead_strip\t");
LABEL_3:
      v5 = 1;
      sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
      sub_E4D880(a1);
      return v5;
    case 19:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.symbol_resolver\t");
      goto LABEL_3;
    case 20:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.alt_entry\t");
      goto LABEL_3;
    case 21:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.private_extern\t");
      goto LABEL_3;
    case 22:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.protected\t");
      goto LABEL_3;
    case 23:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.reference\t");
      goto LABEL_3;
    case 24:
      sub_904010(*(_QWORD *)(a1 + 304), *(const char **)(*(_QWORD *)(a1 + 312) + 296LL));
      goto LABEL_3;
    case 25:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.weak_definition\t");
      goto LABEL_3;
    case 26:
      sub_904010(*(_QWORD *)(a1 + 304), *(const char **)(*(_QWORD *)(a1 + 312) + 304LL));
      goto LABEL_3;
    case 27:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.weak_def_can_be_hidden\t");
      goto LABEL_3;
    case 28:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.weak_anti_dep\t");
      goto LABEL_3;
    case 29:
      sub_904010(*(_QWORD *)(a1 + 304), "\t.memtag\t");
      goto LABEL_3;
    default:
      goto LABEL_3;
  }
}
