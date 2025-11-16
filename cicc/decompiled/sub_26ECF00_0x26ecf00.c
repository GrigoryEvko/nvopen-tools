// Function: sub_26ECF00
// Address: 0x26ecf00
//
void __fastcall sub_26ECF00(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rax

  v2 = *(__int64 **)(a2 + 8);
  v3 = &v2[*(unsigned int *)(a2 + 16)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    sub_26ECD90(a1, *(_QWORD *)(v4 + 8));
  }
}
