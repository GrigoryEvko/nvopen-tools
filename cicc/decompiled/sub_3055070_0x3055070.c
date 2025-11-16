// Function: sub_3055070
// Address: 0x3055070
//
__int64 __fastcall sub_3055070(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax
  char v6; // si
  unsigned __int64 v7; // rdx
  __int64 v8; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 )
  {
    v4 = *a2;
    while ( 1 )
    {
      v7 = *(_QWORD *)(v3 + 32);
      if ( v7 > v4 || v7 == v4 && *((_DWORD *)a2 + 2) < *(_DWORD *)(v3 + 40) )
      {
        v5 = *(_QWORD *)(v3 + 16);
        v6 = 1;
        if ( !v5 )
        {
LABEL_9:
          if ( v6 )
            goto LABEL_10;
LABEL_12:
          if ( v7 < v4 || v7 == v4 && *(_DWORD *)(v3 + 40) < *((_DWORD *)a2 + 2) )
            return 0;
          return v3;
        }
      }
      else
      {
        v5 = *(_QWORD *)(v3 + 24);
        v6 = 0;
        if ( !v5 )
          goto LABEL_9;
      }
      v3 = v5;
    }
  }
  v3 = a1 + 8;
LABEL_10:
  if ( *(_QWORD *)(a1 + 24) != v3 )
  {
    v8 = sub_220EF80(v3);
    v4 = *a2;
    v7 = *(_QWORD *)(v8 + 32);
    v3 = v8;
    goto LABEL_12;
  }
  return 0;
}
