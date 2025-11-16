// Function: sub_1098E50
// Address: 0x1098e50
//
__int64 __fastcall sub_1098E50(__int64 a1, unsigned __int64 a2, signed __int64 a3)
{
  __int16 v3; // dx
  __int64 result; // rax
  void *v5; // rdx
  unsigned __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  __int16 v7; // [rsp+18h] [rbp-18h]

  if ( a3 )
  {
    if ( a2 )
    {
      v6 = sub_F04200(a3, a2);
      v7 = v3;
      sub_D78C90((__int64)&v6, 0);
      return sub_F04D90(a1, v6, v7, 64, 0xAu);
    }
    else
    {
      v5 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 0xCu )
      {
        return sub_CB6200(a1, "<invalid BFI>", 0xDu);
      }
      else
      {
        qmemcpy(v5, "<invalid BFI>", 13);
        *(_QWORD *)(a1 + 32) += 13LL;
        return 0x64696C61766E693CLL;
      }
    }
  }
  else
  {
    result = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) == result )
    {
      return sub_CB6200(a1, (unsigned __int8 *)"0", 1u);
    }
    else
    {
      *(_BYTE *)result = 48;
      ++*(_QWORD *)(a1 + 32);
    }
  }
  return result;
}
