// Function: sub_829C30
// Address: 0x829c30
//
_BOOL8 __fastcall sub_829C30(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __m128i *v4; // r12
  __m128i *v5; // rax
  int v6; // [rsp-1Ch] [rbp-1Ch] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( !*(_QWORD *)(a2 + 8) || !v2 )
    return 0;
  v4 = sub_829BD0(v2);
  v5 = sub_829BD0(*(_QWORD *)(a2 + 8));
  return v4 && v5 && (unsigned int)sub_8DEFB0(v5, v4, 0, &v6) && v6 != 0;
}
