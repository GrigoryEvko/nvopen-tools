// Function: sub_85B8C0
// Address: 0x85b8c0
//
__int64 __fastcall sub_85B8C0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx

  v2 = *(_BYTE *)(a1 + 80);
  switch ( v2 )
  {
    case 3:
      result = sub_72C930();
      *(_QWORD *)(a1 + 88) = result;
      break;
    case 19:
      v5 = sub_87F550(a1, a2);
      v6 = *(_QWORD *)(a1 + 88);
      *(_QWORD *)(v6 + 200) = v5;
      result = *(_QWORD *)(*(_QWORD *)(v5 + 88) + 104LL);
      *(_QWORD *)(v6 + 208) = result;
      break;
    case 2:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 128LL);
      result = (__int64)sub_724D80(0);
      *(_QWORD *)(a1 + 88) = result;
      *(_QWORD *)(result + 128) = v4;
      break;
    default:
      sub_721090();
  }
  return result;
}
