// Function: sub_109CD10
// Address: 0x109cd10
//
char __fastcall sub_109CD10(__int64 *a1, __int64 a2, char a3)
{
  __int64 *v4; // rax
  __int64 v5; // r13
  int v6; // r15d
  int v7; // eax

  sub_B45230(a2, *a1);
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  if ( *(_BYTE *)*v4 == 86 )
  {
    v6 = sub_B45210(*v4);
    v7 = sub_B45210(*a1);
    sub_B45150(a2, v7 | v6);
    LOBYTE(v4) = sub_B451E0(v5);
    if ( !(_BYTE)v4 && !a3 )
    {
      LOBYTE(v4) = sub_98ED60(*(unsigned __int8 **)(v5 - 96), 0, 0, 0, 0);
      if ( !(_BYTE)v4 )
        LOBYTE(v4) = sub_B450D0(a2, 0);
    }
  }
  return (char)v4;
}
