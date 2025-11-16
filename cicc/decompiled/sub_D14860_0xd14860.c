// Function: sub_D14860
// Address: 0xd14860
//
__int64 __fastcall sub_D14860(unsigned __int8 *a1)
{
  int v1; // edx
  unsigned __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v5; // rax

  v1 = *a1;
  if ( (unsigned int)(v1 - 30) > 0xA )
  {
    if ( (_BYTE)v1 == 85 )
    {
      v5 = *((_QWORD *)a1 - 4);
      if ( !v5
        || *(_BYTE *)v5
        || *(_QWORD *)(v5 + 24) != *((_QWORD *)a1 + 10)
        || (*(_BYTE *)(v5 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v5 + 36) - 68) > 3 )
      {
        return sub_B46970(a1);
      }
    }
    else
    {
      v2 = (unsigned int)(v1 - 39);
      if ( (unsigned int)v2 > 0x38 )
        return sub_B46970(a1);
      v3 = 0x100060000000001LL;
      if ( !_bittest64(&v3, v2) )
        return sub_B46970(a1);
    }
  }
  return 1;
}
