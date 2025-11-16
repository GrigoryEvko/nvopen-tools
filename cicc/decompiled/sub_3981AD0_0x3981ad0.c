// Function: sub_3981AD0
// Address: 0x3981ad0
//
void __fastcall sub_3981AD0(__int64 a1, __int64 a2)
{
  char *v2; // rax
  char *v3; // rax
  __int64 v4; // r15
  __int64 v5; // r15
  __int64 v6; // r12
  unsigned __int16 *v7; // rbx
  char *v8; // rax
  char *v9; // rax

  v2 = (char *)sub_14E0540(*(unsigned __int16 *)(a1 + 12));
  sub_397C0C0(a2, *(unsigned __int16 *)(a1 + 12), v2);
  v3 = (char *)sub_14E2A50(*(unsigned __int8 *)(a1 + 14));
  sub_397C0C0(a2, *(unsigned __int8 *)(a1 + 14), v3);
  v4 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = 16 * v4;
    v6 = 0;
    do
    {
      while ( 1 )
      {
        v7 = (unsigned __int16 *)(v6 + *(_QWORD *)(a1 + 16));
        v8 = (char *)sub_14E2A80(*v7);
        sub_397C0C0(a2, *v7, v8);
        v9 = (char *)sub_14E3630(v7[1]);
        sub_397C0C0(a2, v7[1], v9);
        if ( v7[1] == 33 )
          break;
        v6 += 16;
        if ( v5 == v6 )
          goto LABEL_6;
      }
      v6 += 16;
      sub_397C040(a2, *((_QWORD *)v7 + 1), 0);
    }
    while ( v5 != v6 );
  }
LABEL_6:
  sub_397C0C0(a2, 0, "EOM(1)");
  sub_397C0C0(a2, 0, "EOM(2)");
}
