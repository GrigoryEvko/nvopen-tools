// Function: sub_690C20
// Address: 0x690c20
//
void sub_690C20()
{
  __int64 *v0; // rdi
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v4[64]; // [rsp+10h] [rbp-40h] BYREF

  if ( word_4F06418[0] == 67
    && dword_4F04C64 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0 )
  {
    sub_7ADF70(v4, 0);
    v0 = (__int64 *)v4;
    sub_7AE360(v4);
    while ( 1 )
    {
      sub_7B8B50(v0, 0, v1, v2);
      v0 = &v3;
      if ( (unsigned int)sub_869470(&v3) )
        break;
      if ( word_4F06418[0] != 67 )
        return;
    }
    sub_867030(v3);
    sub_7BC000(v4);
  }
}
