// Function: sub_35ED4A0
// Address: 0x35ed4a0
//
__int64 __fastcall sub_35ED4A0(int a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 result; // rax
  _QWORD *v4; // rdx
  _QWORD *v5; // rdx

  switch ( a1 )
  {
    case 1:
    case 5:
      return result;
    case 2:
      v5 = *(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 7u )
      {
        result = sub_CB6200(a2, ".acquire", 8u);
      }
      else
      {
        *v5 = 0x657269757163612ELL;
        *(_QWORD *)(a2 + 32) += 8LL;
        result = 0x657269757163612ELL;
      }
      break;
    case 3:
      v4 = *(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 7u )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)".release", 8u);
      }
      else
      {
        *v4 = 0x657361656C65722ELL;
        *(_QWORD *)(a2 + 32) += 8LL;
        result = 0x657361656C65722ELL;
      }
      break;
    case 4:
      v2 = *(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 7u )
      {
        result = sub_CB6200(a2, ".acq_rel", 8u);
      }
      else
      {
        *v2 = 0x6C65725F7163612ELL;
        *(_QWORD *)(a2 + 32) += 8LL;
        result = 0x6C65725F7163612ELL;
      }
      break;
    default:
      sub_C64ED0("unsupported ordering for nvvm.atomic.rmw", 1u);
  }
  return result;
}
