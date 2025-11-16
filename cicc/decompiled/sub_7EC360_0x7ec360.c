// Function: sub_7EC360
// Address: 0x7ec360
//
void __fastcall sub_7EC360(__int64 a1, __m128i *a2, __int64 *a3)
{
  _QWORD *v4; // r14
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _BYTE *v7; // rax
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  switch ( a2->m128i_i8[0] )
  {
    case 0:
    case 2:
    case 4:
    case 5:
    case 6:
      return;
    case 1:
      sub_7EB190(*a3, a2);
      break;
    case 3:
      if ( (unsigned int)sub_7E3130(*(_QWORD *)(a1 + 120)) )
      {
        v8[0] = 0;
        if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
          sub_7296C0(v8);
        v4 = sub_7E4750(*(_QWORD *)(a1 + 120));
        sub_729730(v8[0]);
        if ( *(_BYTE *)(a1 + 136) > 2u )
        {
          a2->m128i_i8[0] = 2;
          v5 = sub_725A70(2u);
          *a3 = (__int64)v5;
          v6 = v5;
          v5[7] = v4;
          v5[1] = a1;
          v7 = sub_726B30(17);
          *((_QWORD *)v7 + 9) = v6;
          sub_7FCA00(v7);
        }
        else
        {
          a2->m128i_i8[0] = 1;
          *a3 = (__int64)v4;
        }
      }
      break;
    default:
      sub_721090();
  }
}
