// Function: sub_1055C50
// Address: 0x1055c50
//
void __fastcall sub_1055C50(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *((unsigned int *)a1 + 38);
  sub_C7D6A0(a1[17], v1, 8);
  if ( *((_BYTE *)a1 + 92) )
  {
    if ( *((_BYTE *)a1 + 28) )
      return;
LABEL_5:
    _libc_free(a1[1], v1);
    return;
  }
  _libc_free(a1[9], v1);
  if ( !*((_BYTE *)a1 + 28) )
    goto LABEL_5;
}
