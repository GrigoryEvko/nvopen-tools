// Function: sub_19C89A0
// Address: 0x19c89a0
//
__int64 __fastcall sub_19C89A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4FB65F4);
  sub_1636A40(a2, (__int64)&unk_5051F8C);
  sub_1636A40(a2, (__int64)&unk_4F9920C);
  sub_1636A40(a2, (__int64)&unk_4FB66D8);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F96DB4;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v6;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v6 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1636A40(a2, (__int64)&unk_4F99CB0);
}
