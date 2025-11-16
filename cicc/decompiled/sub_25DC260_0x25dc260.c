// Function: sub_25DC260
// Address: 0x25dc260
//
bool __fastcall sub_25DC260(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v5; // rsi
  unsigned int v7; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_F02DB0(&v7, qword_4FF1668, 0x64u);
  v2 = sub_FDD860(a2, *(_QWORD *)(a1 + 40));
  v3 = sub_B491C0(a1);
  v5 = *(_QWORD *)(v3 + 80);
  if ( v5 )
    v5 -= 24;
  v8[0] = sub_FDD860(a2, v5);
  return sub_1098D20(v8, v7) > v2;
}
