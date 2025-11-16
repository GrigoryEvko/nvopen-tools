// Function: sub_373DD80
// Address: 0x373dd80
//
void __fastcall sub_373DD80(__int64 a1, _QWORD *a2, __int64 a3)
{
  _BYTE *v3; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax

  if ( a2 )
  {
    v3 = (_BYTE *)a2[1];
    if ( v3 )
    {
      if ( *a2 && *v3 == 18 )
      {
        sub_373D2D0(a1, (__int64)a2, a3);
      }
      else
      {
        if ( sub_3735ED0(a1, (__int64)a2) )
          return;
        v5 = sub_373D7E0(a1, (__int64)a2);
        *(_QWORD *)(v5 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
        v6 = *(_QWORD **)(a3 + 32);
        if ( v6 )
        {
          *(_QWORD *)v5 = *v6;
          **(_QWORD **)(a3 + 32) = v5 & 0xFFFFFFFFFFFFFFFBLL;
        }
        *(_QWORD *)(a3 + 32) = v5;
      }
      sub_373DE40(a1, a2);
    }
  }
}
