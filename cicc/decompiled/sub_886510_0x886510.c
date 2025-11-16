// Function: sub_886510
// Address: 0x886510
//
void __fastcall sub_886510(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rax
  int v3; // r13d
  int v4; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( !dword_4F5FFBC )
  {
    v4 = 0;
    v1 = qword_4D049B8;
    v2 = *(_QWORD *)(a1 + 8);
    v3 = dword_4F04C5C;
    dword_4F04C5C = 0;
    qword_4D049B8[6] = v2;
    *(_QWORD *)(a1 + 24) = v1;
    sub_885620((__int64)v1, 0, &v4);
    sub_881ED0((__int64)v1, dword_4F04C64, v4);
    dword_4F04C5C = v3;
    dword_4F5FFBC = 1;
  }
}
