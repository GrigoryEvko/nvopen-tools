// Function: sub_38D34B0
// Address: 0x38d34b0
//
__int64 __fastcall sub_38D34B0(__int64 a1, __int64 a2, char a3, __int64 a4, char a5)
{
  _DWORD *v5; // r14
  int v7; // ecx
  __int64 result; // rax

  v5 = (_DWORD *)(a1 + 696);
  *(_QWORD *)(a1 + 688) = a4;
  *(_BYTE *)(a1 + 684) = a3;
  *(_QWORD *)a1 = 257;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  sub_2240AE0((unsigned __int64 *)(a1 + 696), (unsigned __int64 *)a2);
  *(_DWORD *)(a1 + 728) = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 732) = *(_DWORD *)(a2 + 36);
  *(_DWORD *)(a1 + 736) = *(_DWORD *)(a2 + 40);
  v7 = *(_DWORD *)(a2 + 44);
  *(_DWORD *)(a1 + 740) = v7;
  *(_DWORD *)(a1 + 744) = *(_DWORD *)(a2 + 48);
  result = *(unsigned int *)(a2 + 52);
  *(_DWORD *)(a1 + 748) = result;
  switch ( (int)result )
  {
    case 0:
      sub_16BD130("Cannot initialize MC for unknown object file format.", 1u);
    case 1:
      if ( v7 != 15 )
        sub_16BD130("Cannot initialize MC for non-Windows COFF object files.", 1u);
      *(_DWORD *)(a1 + 680) = 2;
      result = sub_38D2820(a1, (__int64)v5);
      break;
    case 2:
      *(_DWORD *)(a1 + 680) = 1;
      result = sub_38D17C0(a1, (__int64)v5, a5);
      break;
    case 3:
      *(_DWORD *)(a1 + 680) = 0;
      result = sub_38D08C0(a1, v5);
      break;
    case 4:
      *(_DWORD *)(a1 + 680) = 3;
      result = sub_38D3050((_QWORD *)a1);
      break;
    default:
      return result;
  }
  return result;
}
