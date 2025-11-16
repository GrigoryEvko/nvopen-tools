// Function: sub_80A450
// Address: 0x80a450
//
void __fastcall sub_80A450(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13

  if ( a2 == 7 )
  {
    v3 = *(_QWORD *)(a1 + 120);
    goto LABEL_5;
  }
  if ( !(_DWORD)qword_4F077B4 && (!HIDWORD(qword_4F077B4) || qword_4F077A8 <= 0x1116Fu) || *(_BYTE *)(a1 + 174) != 3 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 160LL);
LABEL_5:
    if ( v3 )
    {
      v4 = sub_8D21C0(v3);
      if ( !(unsigned int)sub_8D2600(v4)
        && !(unsigned int)sub_8D2780(v4)
        && !(unsigned int)sub_8D2A90(v4)
        && !(unsigned int)sub_8D4C80(v4) )
      {
        byte_4F18B8C = a2;
        qword_4F18B90 = a1;
        sub_809D70(a1, a2, 1);
        sub_8D9600(v4, sub_80A590, 27);
        sub_809D70(a1, a2, 0);
        qword_4F18B90 = 0;
        byte_4F18B8C = 0;
        if ( qword_4F18BA0 )
        {
          sub_7252B0(qword_4F18BA0);
          qword_4F18BA0 = 0;
        }
      }
    }
  }
}
