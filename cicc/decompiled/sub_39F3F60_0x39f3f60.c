// Function: sub_39F3F60
// Address: 0x39f3f60
//
void __fastcall sub_39F3F60(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax

  switch ( a2 )
  {
    case 0:
      sub_39F3EB0(a1, 1);
      break;
    case 1:
      sub_39F3EB0(a1, 2);
      break;
    case 2:
      sub_39F3EB0(a1, 3);
      break;
    case 3:
      sub_39F3EB0(a1, 4);
      break;
    case 4:
      v2 = *(_QWORD *)(*(_QWORD *)(a1 + 264) + 112LL);
      v3 = sub_38BFA60(*(_QWORD *)(a1 + 8), 1);
      *(_QWORD *)(v2 - 8) = v3;
      sub_39F3BB0(a1, v3, 0);
      break;
    default:
      return;
  }
}
