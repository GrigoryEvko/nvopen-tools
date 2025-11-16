// Function: sub_71B620
// Address: 0x71b620
//
__int64 __fastcall sub_71B620(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rcx
  __int64 v4; // r12

  v1 = sub_8D71D0(a1);
  v2 = sub_73C570(v1, 1, -1);
  v4 = sub_735FB0(v2, 3, 0xFFFFFFFFLL, v3);
  *(_QWORD *)(v4 + 256) = v2;
  *(_BYTE *)(v4 + 89) |= 1u;
  *(_QWORD *)(v4 + 168) |= 0x100008000uLL;
  sub_72EE40(v4, 7, qword_4F04C50);
  return v4;
}
