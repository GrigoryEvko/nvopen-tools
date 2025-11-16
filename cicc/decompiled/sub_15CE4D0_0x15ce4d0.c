// Function: sub_15CE4D0
// Address: 0x15ce4d0
//
void __fastcall sub_15CE4D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rax
  char *v4[6]; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 != a2 )
  {
    v4[0] = (char *)a1;
    v3 = sub_15CBEB0(*(_QWORD **)(v2 + 24), *(_QWORD *)(v2 + 32), (__int64 *)v4);
    sub_15CDF70(*(_QWORD *)(a1 + 8) + 24LL, v3);
    v4[0] = (char *)a1;
    *(_QWORD *)(a1 + 8) = a2;
    sub_15CE4A0(a2 + 24, v4);
    sub_15CC3F0(a1);
  }
}
