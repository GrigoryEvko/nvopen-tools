// Function: sub_7E90E0
// Address: 0x7e90e0
//
void __fastcall sub_7E90E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  char v4; // al
  _BYTE *v5; // rbx
  __int64 v6; // rbx
  _QWORD *v7; // rax

  v2 = sub_7F9300();
  *(_QWORD *)(a1 + 80) = v2;
  v3 = v2;
  v4 = *(_BYTE *)(a1 + 49);
  if ( (v4 & 0x40) != 0 )
  {
    v5 = *(_BYTE **)(a1 + 96);
    if ( v5 )
    {
      sub_7FAED0(*(_QWORD *)(a1 + 96));
      v4 = *(_BYTE *)(a1 + 49);
      if ( v4 >= 0 && (*v5 & 2) == 0 )
        goto LABEL_5;
    }
    else if ( v4 >= 0 )
    {
      goto LABEL_5;
    }
    sub_733B20((_QWORD *)a1);
    return;
  }
LABEL_5:
  if ( (v4 & 4) != 0 )
  {
    v6 = *(_QWORD *)(a1 + 80);
    v7 = sub_72BA30(5u);
    *(_QWORD *)(v6 + 88) = sub_7E7CA0((__int64)v7);
    if ( a2 )
      sub_7FA230(v3, a2);
  }
}
