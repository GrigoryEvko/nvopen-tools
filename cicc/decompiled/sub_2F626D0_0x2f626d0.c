// Function: sub_2F626D0
// Address: 0x2f626d0
//
__int64 __fastcall sub_2F626D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax

  if ( !*(_BYTE *)(a1 + 692) )
  {
LABEL_8:
    sub_C8CC70(a1 + 664, a2, (__int64)a3, a4, a5, a6);
    goto LABEL_6;
  }
  v6 = *(__int64 **)(a1 + 672);
  a4 = *(unsigned int *)(a1 + 684);
  a3 = &v6[a4];
  if ( v6 == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 680) )
    {
      *(_DWORD *)(a1 + 684) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 664);
      goto LABEL_6;
    }
    goto LABEL_8;
  }
  while ( a2 != *v6 )
  {
    if ( a3 == ++v6 )
      goto LABEL_7;
  }
LABEL_6:
  sub_2FAD510(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), a2, 0);
  return sub_2E88E20(a2);
}
