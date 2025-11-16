// Function: sub_127CDE0
// Address: 0x127cde0
//
void __fastcall sub_127CDE0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rdi

  v3 = *(_QWORD *)(a1 + 40);
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  while ( v3 )
  {
    sub_127AA60(*(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4, 40);
  }
  v5 = *(_QWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 32;
  *(_QWORD *)(a1 + 64) = 0;
  while ( v5 )
  {
    sub_127A890(*(_QWORD *)(v5 + 24));
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
    j_j___libc_free_0(v6, 48);
  }
  v7 = *(_QWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = a1 + 80;
  *(_QWORD *)(a1 + 104) = a1 + 80;
  *(_QWORD *)(a1 + 112) = 0;
  while ( v7 )
  {
    sub_127AA60(*(_QWORD *)(v7 + 24));
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v8, 40);
  }
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = a1 + 128;
  *(_QWORD *)(a1 + 152) = a1 + 128;
  *(_QWORD *)(a1 + 160) = 0;
  sub_127CD40(a1, a2);
}
