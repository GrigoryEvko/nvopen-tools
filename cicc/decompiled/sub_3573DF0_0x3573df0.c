// Function: sub_3573DF0
// Address: 0x3573df0
//
void __fastcall sub_3573DF0(_QWORD *a1)
{
  sub_C7D6A0(a1[17], 16LL * *((unsigned int *)a1 + 38), 8);
  if ( *((_BYTE *)a1 + 92) )
  {
    if ( *((_BYTE *)a1 + 28) )
      return;
LABEL_5:
    _libc_free(a1[1]);
    return;
  }
  _libc_free(a1[9]);
  if ( !*((_BYTE *)a1 + 28) )
    goto LABEL_5;
}
