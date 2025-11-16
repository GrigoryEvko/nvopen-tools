// Function: sub_35F4DA0
// Address: 0x35f4da0
//
__int64 __fastcall sub_35F4DA0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // rdx

  v4 = *(_QWORD *)(a4 + 24);
  v5 = (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 1) == 0;
  v6 = *(_QWORD *)(a4 + 32);
  if ( v5 )
  {
    if ( (unsigned __int64)(v4 - v6) <= 8 )
    {
      return sub_CB6200(a4, ".wait::ld", 9u);
    }
    else
    {
      *(_BYTE *)(v6 + 8) = 100;
      *(_QWORD *)v6 = 0x6C3A3A746961772ELL;
      *(_QWORD *)(a4 + 32) += 9LL;
      return 0x6C3A3A746961772ELL;
    }
  }
  else if ( (unsigned __int64)(v4 - v6) <= 8 )
  {
    return sub_CB6200(a4, ".wait::st", 9u);
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 116;
    *(_QWORD *)v6 = 0x733A3A746961772ELL;
    *(_QWORD *)(a4 + 32) += 9LL;
    return 0x733A3A746961772ELL;
  }
}
