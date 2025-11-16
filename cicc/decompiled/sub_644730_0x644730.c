// Function: sub_644730
// Address: 0x644730
//
void __fastcall sub_644730(__int64 *a1)
{
  __int64 *v1; // rcx
  char v2; // al
  unsigned __int8 v3; // dl
  __int64 v4; // rdi

  if ( !a1 )
    return;
  v1 = 0;
  do
  {
    v2 = *((_BYTE *)a1 + 9);
    v3 = *((_BYTE *)a1 + 8);
    if ( v2 == 1 || v2 == 4 )
    {
      if ( v3 == 1 )
        goto LABEL_5;
      if ( !HIDWORD(qword_4F077B4) )
      {
        v1 = a1;
        v4 = 8;
LABEL_11:
        sub_684AA0(v4, 1866, v1 + 7);
        return;
      }
    }
    if ( v3 >= 2u )
      v1 = a1;
LABEL_5:
    a1 = (__int64 *)*a1;
  }
  while ( a1 );
  v4 = 5;
  if ( v1 )
    goto LABEL_11;
}
