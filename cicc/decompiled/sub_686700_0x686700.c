// Function: sub_686700
// Address: 0x686700
//
void __fastcall sub_686700(FILE **a1, int a2)
{
  FILE *v2; // r8
  int v3; // [rsp-1Ch] [rbp-1Ch]

  v2 = *a1;
  if ( *a1 )
  {
    *a1 = 0;
    if ( (unsigned int)sub_720ED0(v2) )
      sub_6866A0(a2, v3);
  }
}
