// Function: sub_3982640
// Address: 0x3982640
//
__int64 __fastcall sub_3982640(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 result; // rax
  __int16 v4; // r8

  switch ( a3 )
  {
    case 16:
      v4 = sub_3971A70(a2);
      result = 4;
      if ( v4 == 2 )
        result = *(unsigned int *)(*(_QWORD *)(a2 + 240) + 8LL);
      break;
    case 17:
      result = 1;
      break;
    case 18:
      result = 2;
      break;
    case 19:
      result = 4;
      break;
    case 20:
      result = 8;
      break;
    case 21:
      result = sub_3946290(*(unsigned int *)(*(_QWORD *)a1 + 16LL));
      break;
  }
  return result;
}
