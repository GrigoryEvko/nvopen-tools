// Function: sub_2307CE0
// Address: 0x2307ce0
//
void __fastcall sub_2307CE0(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi

  v2 = a1[15];
  *a1 = &unk_4A0D138;
  while ( v2 )
  {
    sub_2307A70(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  v4 = a1[9];
  while ( v4 )
  {
    sub_23076D0(*(_QWORD *)(v4 + 24));
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    j_j___libc_free_0(v5);
  }
  v6 = a1[3];
  while ( v6 )
  {
    sub_23078A0(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
