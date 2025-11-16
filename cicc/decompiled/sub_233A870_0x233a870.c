// Function: sub_233A870
// Address: 0x233a870
//
void __fastcall sub_233A870(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi

  v2 = a1[14];
  while ( v2 )
  {
    sub_2307A70(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = a1[8];
  while ( v4 )
  {
    sub_23076D0(*(_QWORD *)(v4 + 24));
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    j_j___libc_free_0(v5);
  }
  v6 = a1[2];
  while ( v6 )
  {
    sub_23078A0(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
}
