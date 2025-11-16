// Function: sub_310BF50
// Address: 0x310bf50
//
__int64 *__fastcall sub_310BF50(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  __int64 v6; // r13
  __int64 *result; // rax
  _QWORD *v10; // r10
  _QWORD *v11; // r14
  __int64 *v12; // r8
  __int64 v13; // rdi
  bool v14; // al
  _QWORD *v15; // [rsp+8h] [rbp-88h]
  __int64 *v16; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+20h] [rbp-70h] BYREF
  __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v20[3]; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v21; // [rsp+48h] [rbp-48h]
  __int64 *v22; // [rsp+50h] [rbp-40h]
  __int64 v23; // [rsp+58h] [rbp-38h]

  v6 = a2;
  sub_310A860((__int64 *)v20, a1, a2, a3);
  if ( a2 == a3 )
  {
    *a4 = v23;
    result = v22;
    *a5 = (__int64)v22;
  }
  else if ( sub_D968A0(a2) )
  {
    result = v22;
    *a4 = (__int64)v22;
    *a5 = (__int64)result;
  }
  else if ( sub_D96900(a3) )
  {
    *a4 = a2;
    result = v22;
    *a5 = (__int64)v22;
  }
  else if ( *(_WORD *)(a3 + 24) == 6 )
  {
    *a4 = a2;
    v10 = *(_QWORD **)(a3 + 32);
    v15 = &v10[*(_QWORD *)(a3 + 40)];
    if ( v15 == v10 )
    {
LABEL_16:
      result = v22;
      *a5 = (__int64)v22;
    }
    else
    {
      v11 = *(_QWORD **)(a3 + 32);
      v12 = &v19;
      while ( 1 )
      {
        v16 = v12;
        sub_310BF50(a1, a2, *v11, &v18, v12);
        v13 = v19;
        *a4 = v18;
        v14 = sub_D968A0(v13);
        v12 = v16;
        if ( !v14 )
          break;
        if ( v15 == ++v11 )
          goto LABEL_16;
        a2 = *a4;
      }
      *a4 = (__int64)v22;
      *a5 = v6;
      return a5;
    }
  }
  else
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
        sub_310A630((__int64)v20, a2);
        break;
      case 1:
        sub_310A850(v20, a2);
        break;
      case 2:
      case 3:
      case 4:
      case 7:
      case 9:
      case 0xA:
      case 0xB:
      case 0xC:
      case 0xD:
      case 0xE:
      case 0xF:
      case 0x10:
        break;
      case 5:
        sub_310C4F0(v20, a2);
        break;
      case 6:
        sub_310B7A0(v20, a2);
        break;
      case 8:
        sub_310C140(v20, a2);
        break;
      default:
        BUG();
    }
    *a4 = (__int64)v20[2];
    result = v21;
    *a5 = (__int64)v21;
  }
  return result;
}
