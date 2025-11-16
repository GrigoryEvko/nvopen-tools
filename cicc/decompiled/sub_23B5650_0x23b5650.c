// Function: sub_23B5650
// Address: 0x23b5650
//
__int64 __fastcall sub_23B5650(__int64 *a1)
{
  __int64 v2; // rdi
  void *(*v3)(); // rax
  void *v4; // rax

  if ( !a1 )
    return 0;
  v2 = *a1;
  if ( v2
    && ((v3 = *(void *(**)())(*(_QWORD *)v2 + 24LL), v3 != sub_23AE340) ? (v4 = v3()) : (v4 = &unk_4CDFBF8),
        v4 == &unk_4C5D160) )
  {
    return *a1 + 8;
  }
  else
  {
    return 0;
  }
}
