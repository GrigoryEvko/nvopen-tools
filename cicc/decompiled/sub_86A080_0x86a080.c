// Function: sub_86A080
// Address: 0x86a080
//
__int64 __fastcall sub_86A080(_QWORD *a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  int v4; // esi
  _QWORD **v5; // rax
  _QWORD **v6; // rax
  __int64 v7; // rax

  v1 = (_QWORD *)a1[1];
  if ( v1 )
  {
    v2 = *a1;
    if ( *a1 )
    {
      *v1 = v2;
      *(_QWORD *)(v2 + 8) = a1[1];
      return sub_869F90(a1);
    }
    v4 = dword_4F04C64;
    if ( dword_4F04C64 != -1 )
    {
      if ( dword_4F04C64 >= 0 )
      {
        v5 = (_QWORD **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
        do
        {
          if ( *v5 == a1 )
            break;
          --v4;
          v5 -= 97;
        }
        while ( v4 != -1 );
        return sub_869FD0(a1, v4);
      }
LABEL_20:
      v4 = -1;
      return sub_869FD0(a1, v4);
    }
    *v1 = 0;
  }
  else
  {
    v4 = dword_4F04C64;
    if ( dword_4F04C64 != -1 )
    {
      if ( dword_4F04C64 >= 0 )
      {
        v6 = (_QWORD **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
        do
        {
          if ( *v6 == a1 )
            break;
          --v4;
          v6 -= 97;
        }
        while ( v4 != -1 );
        return sub_869FD0(a1, v4);
      }
      goto LABEL_20;
    }
    v7 = *a1;
    *(_QWORD *)(qword_4F07288 + 256) = *a1;
    if ( v7 )
      *(_QWORD *)(v7 + 8) = 0;
  }
  return sub_869F90(a1);
}
