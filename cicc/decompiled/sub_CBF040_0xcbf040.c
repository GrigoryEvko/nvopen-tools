// Function: sub_CBF040
// Address: 0xcbf040
//
__int64 __fastcall sub_CBF040(_BYTE *a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rcx
  _BYTE *v4; // rax
  char v5; // dl
  _BYTE *v6; // rdx

  if ( a3 )
  {
    v3 = (__int64)&a1[a3 - 1];
    v4 = a2;
    while ( 1 )
    {
      v6 = v4++;
      if ( a1 == (_BYTE *)v3 )
        break;
      v5 = *(v4 - 1);
      *a1++ = v5;
      if ( !v5 )
        return v4 - a2 - 1;
    }
    *a1 = 0;
    v4 = v6 + 1;
    if ( *v6 )
      goto LABEL_8;
    return v4 - a2 - 1;
  }
  else
  {
    v4 = a2;
    do
LABEL_8:
      ++v4;
    while ( *(v4 - 1) );
    return v4 - a2 - 1;
  }
}
