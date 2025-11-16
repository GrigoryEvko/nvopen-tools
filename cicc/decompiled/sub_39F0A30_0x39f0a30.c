// Function: sub_39F0A30
// Address: 0x39f0a30
//
void __fastcall sub_39F0A30(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 *v4; // rsi

  v1 = *(_QWORD *)(a1 + 264);
  v2 = *(__int64 **)(v1 + 2104);
  v3 = *(__int64 **)(v1 + 2112);
  while ( v3 != v2 )
  {
    sub_39F07E0(a1, v2);
    v4 = v2 + 1;
    v2 += 3;
    sub_39F07E0(a1, v4);
  }
}
