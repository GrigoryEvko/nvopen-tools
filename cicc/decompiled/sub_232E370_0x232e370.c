// Function: sub_232E370
// Address: 0x232e370
//
__int64 __fastcall sub_232E370(const void *a1, size_t a2)
{
  __int64 v2; // rbx
  __int64 v4; // [rsp+64h] [rbp-4Ch]

  v2 = unk_5033F08;
  if ( sub_9691B0(a1, a2, "O0", 2) )
    return v2;
  v2 = unk_5033F00;
  if ( sub_9691B0(a1, a2, "O1", 2) )
    return v2;
  v2 = unk_5033EF8;
  if ( sub_9691B0(a1, a2, "O2", 2) )
    return v2;
  v2 = unk_5033EF0;
  if ( sub_9691B0(a1, a2, "O3", 2) )
    return v2;
  v2 = unk_5033EE8;
  if ( sub_9691B0(a1, a2, "Os", 2) )
    return v2;
  v2 = qword_5033EE0;
  v4 = qword_5033EE0;
  if ( sub_9691B0(a1, a2, "Oz", 2) )
    return v2;
  return v4;
}
