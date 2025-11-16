// Function: sub_35F55F0
// Address: 0x35f55f0
//
__int64 __fastcall sub_35F55F0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  _QWORD *v4; // rdx
  __int64 result; // rax

  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 1) != 0 )
  {
    v4 = *(_QWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v4 <= 7u )
    {
      return sub_CB6200(a4, (unsigned __int8 *)".aligned", 8u);
    }
    else
    {
      *v4 = 0x64656E67696C612ELL;
      *(_QWORD *)(a4 + 32) += 8LL;
      return 0x64656E67696C612ELL;
    }
  }
  return result;
}
