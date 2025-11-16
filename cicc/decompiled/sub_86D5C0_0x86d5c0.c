// Function: sub_86D5C0
// Address: 0x86d5c0
//
void __fastcall sub_86D5C0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r12

  v2 = **(_QWORD **)(a1 + 72);
  if ( *(_BYTE *)(a1 + 40) == 7 )
  {
    v5 = *(_QWORD *)(v2 + 96);
    v6 = sub_86B2C0(3);
    *(_QWORD *)(v6 + 40) = a1;
    *(_QWORD *)(v6 + 24) = *a2;
    sub_86CBE0(v6);
    *(_QWORD *)(v2 + 96) = v6;
    if ( v5 )
      sub_86C540(v6, v5, 1);
  }
  else
  {
    v3 = (_QWORD *)sub_86B2C0(2);
    v4 = *a2;
    v3[5] = a1;
    v3[3] = v4;
    sub_86CBE0((__int64)v3);
    if ( (*(_BYTE *)(v2 + 81) & 2) != 0 )
    {
      sub_86C540(*(_QWORD *)(v2 + 96), (__int64)v3, 0);
    }
    else
    {
      v3[6] = *(_QWORD *)(v2 + 96);
      *(_QWORD *)(v2 + 96) = v3;
    }
  }
}
