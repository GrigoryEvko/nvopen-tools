// Function: sub_B5B280
// Address: 0xb5b280
//
__int64 __fastcall sub_B5B280(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v6; // [rsp+18h] [rbp-48h] BYREF
  __int64 v7; // [rsp+28h] [rbp-38h] BYREF
  __int64 v8; // [rsp+30h] [rbp-30h] BYREF
  __int64 v9; // [rsp+38h] [rbp-28h]
  __int64 v10; // [rsp+40h] [rbp-20h]

  v6 = a3;
  if ( (unsigned int)a2 > 0x1EC )
  {
LABEL_9:
    v7 = *(_QWORD *)(*a4 + 8LL);
    a2 = (unsigned int)a2;
    if ( sub_B5B000(a2) )
    {
      a2 = (unsigned int)a2;
      v8 = sub_B5B240(a2);
      v7 = *(_QWORD *)(a4[(unsigned int)v8] + 8LL);
    }
    return sub_B6E160(a1, a2, &v7, 1);
  }
  if ( (unsigned int)a2 <= 0x19E )
  {
    switch ( (_DWORD)a2 )
    {
      case 0xA7:
        v8 = v6;
        v9 = *(_QWORD *)(*a4 + 8LL);
        v10 = *(_QWORD *)(a4[1] + 8LL);
        return sub_B6E160(a1, 167, &v8, 3);
      case 0xA8:
        v8 = *(_QWORD *)(*a4 + 8LL);
        v9 = *(_QWORD *)(a4[1] + 8LL);
        v10 = *(_QWORD *)(a4[2] + 8LL);
        return sub_B6E160(a1, 168, &v8, 3);
      case 0xA5:
        return sub_B6E160(a1, 165, &v6, 1);
    }
    goto LABEL_9;
  }
  switch ( (int)a2 )
  {
    case 415:
    case 425:
    case 426:
    case 427:
    case 428:
    case 435:
    case 437:
    case 439:
    case 449:
    case 473:
    case 475:
    case 483:
    case 486:
    case 492:
      v8 = v6;
      v9 = *(_QWORD *)(*a4 + 8LL);
      result = sub_B6E160(a1, a2, &v8, 2);
      break;
    case 433:
      v8 = v6;
      v9 = *(_QWORD *)(*a4 + 8LL);
      result = sub_B6E160(a1, 433, &v8, 2);
      break;
    case 436:
      v8 = *(_QWORD *)(*a4 + 8LL);
      result = sub_B6E160(a1, 436, &v8, 1);
      break;
    case 438:
      v8 = v6;
      v9 = *(_QWORD *)(*a4 + 8LL);
      result = sub_B6E160(a1, 438, &v8, 2);
      break;
    case 443:
    case 472:
      v8 = *(_QWORD *)(a4[1] + 8LL);
      result = sub_B6E160(a1, a2, &v8, 1);
      break;
    case 470:
      v8 = *(_QWORD *)(*a4 + 8LL);
      v9 = *(_QWORD *)(a4[1] + 8LL);
      result = sub_B6E160(a1, 470, &v8, 2);
      break;
    case 481:
      v8 = *(_QWORD *)(*a4 + 8LL);
      v9 = *(_QWORD *)(a4[1] + 8LL);
      result = sub_B6E160(a1, 481, &v8, 2);
      break;
    default:
      goto LABEL_9;
  }
  return result;
}
