// Function: sub_35ED820
// Address: 0x35ed820
//
unsigned __int64 __fastcall sub_35ED820(int a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 32);
  switch ( a1 )
  {
    case 0:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)".f32", 4u);
      }
      else
      {
        *(_DWORD *)v2 = 842229294;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 1:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 5 )
      {
        result = sub_CB6200(a2, ".f16x2", 6u);
      }
      else
      {
        *(_DWORD *)v2 = 909207086;
        *(_WORD *)(v2 + 4) = 12920;
        *(_QWORD *)(a2 + 32) += 6LL;
      }
      break;
    case 2:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 6 )
      {
        result = sub_CB6200(a2, ".e4m3x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 1832150318;
        *(_WORD *)(v2 + 4) = 30771;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      break;
    case 3:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 6 )
      {
        result = sub_CB6200(a2, ".e5m2x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 1832215854;
        *(_WORD *)(v2 + 4) = 30770;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      break;
    case 4:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 6 )
      {
        result = sub_CB6200(a2, ".bf16x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 828793390;
        *(_WORD *)(v2 + 4) = 30774;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      break;
    case 5:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 6 )
      {
        result = sub_CB6200(a2, ".e2m1x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 1832019246;
        *(_WORD *)(v2 + 4) = 30769;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      break;
    case 6:
      result = *(_QWORD *)(a2 + 24) - v2;
      if ( result <= 6 )
      {
        result = sub_CB6200(a2, ".e2m3x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 1832019246;
        *(_WORD *)(v2 + 4) = 30771;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
      break;
    case 7:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v2) <= 6 )
      {
        result = sub_CB6200(a2, ".e3m2x2", 7u);
      }
      else
      {
        *(_DWORD *)v2 = 1832084782;
        *(_WORD *)(v2 + 4) = 30770;
        *(_BYTE *)(v2 + 6) = 50;
        *(_QWORD *)(a2 + 32) += 7LL;
        result = 30770;
      }
      break;
    case 8:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v2) <= 7 )
      {
        result = sub_CB6200(a2, ".ue8m0x2", 8u);
      }
      else
      {
        *(_QWORD *)v2 = 0x3278306D3865752ELL;
        *(_QWORD *)(a2 + 32) += 8LL;
        result = 0x3278306D3865752ELL;
      }
      break;
    default:
      sub_C64ED0("Invalid Src/Dst Type in cvt_packfloat Intrinsic.", 1u);
  }
  return result;
}
