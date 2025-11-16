// Function: sub_88E6E0
// Address: 0x88e6e0
//
void __fastcall sub_88E6E0(__int64 *a1, unsigned int a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rdi
  char v4; // al
  __int64 v5; // r13

  if ( a1 )
  {
    v2 = a1;
    while ( 1 )
    {
      v4 = *((_BYTE *)v2 + 8);
      if ( !v4 )
        break;
      if ( v4 == 1 && (v2[3] & 1) == 0 && (v5 = v2[4]) != 0 )
      {
        *(_QWORD *)(v5 + 128) = sub_8E3240(*(_QWORD *)(v5 + 128), a2);
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
      else
      {
LABEL_5:
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
    }
    v3 = v2[4];
    if ( v3 )
      v2[4] = sub_8E3240(v3, a2);
    goto LABEL_5;
  }
}
