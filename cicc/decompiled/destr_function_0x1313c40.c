// Function: destr_function
// Address: 0x1313c40
//
void __fastcall destr_function(_BYTE *a1)
{
  _BYTE *v1; // rdx

  switch ( a1[816] )
  {
    case 0:
    case 1:
    case 3:
    case 5:
      nullsub_502();
      sub_1304740((__int64)a1);
      sub_1304790((__int64)a1);
      sub_1311F30((__int64)a1);
      nullsub_507(a1 + 2616);
      a1[1] = 1;
      sub_1313AB0((__int64)a1, 4u);
      v1 = (_BYTE *)(__readfsqword(0) - 2664);
      if ( a1 != v1 )
        qmemcpy(v1, a1, 0xA48u);
      if ( pthread_setspecific(key, (const void *)(__readfsqword(0) - 2664)) )
      {
        sub_130AA40("<jemalloc>: Error setting tsd.\n");
        if ( byte_4F969A5[0] )
          abort();
      }
      return;
    case 2:
    case 4:
    case 6:
      return;
  }
}
