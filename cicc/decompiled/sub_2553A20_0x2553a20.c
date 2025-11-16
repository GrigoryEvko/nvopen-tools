// Function: sub_2553A20
// Address: 0x2553a20
//
__int64 __fastcall sub_2553A20(__int64 a1, unsigned int a2)
{
  int v2; // eax
  __int64 result; // rax
  const char *v4[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = sub_B49240(a1);
  if ( v2 != 3142 )
  {
    if ( (unsigned int)(v2 - 8259) <= 3 )
      return 1;
LABEL_6:
    sub_2553790(v4, "ompx_aligned_barrier");
    return sub_31402F0(a1, v4);
  }
  result = a2;
  if ( !(_BYTE)a2 )
    goto LABEL_6;
  return result;
}
