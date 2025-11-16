// Function: sub_E85600
// Address: 0xe85600
//
void __fastcall sub_E85600(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // rax

  switch ( a2 )
  {
    case 0u:
      sub_E85550(a1, 1, a3, a4, a5);
      break;
    case 1u:
      sub_E85550(a1, 2, a3, a4, a5);
      break;
    case 2u:
      sub_E85550(a1, 3, a3, a4, a5);
      break;
    case 3u:
      sub_E85550(a1, 4, a3, a4, a5);
      break;
    case 4u:
      v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL) + 208LL);
      v6 = sub_E6C430(*(_QWORD *)(a1 + 8), a2, a3, a4, a5);
      *(_QWORD *)(v5 - 8) = v6;
      sub_E85210(a1, v6, 0);
      break;
    default:
      return;
  }
}
