// Function: sub_2988BD0
// Address: 0x2988bd0
//
__int64 __fastcall sub_2988BD0(_QWORD *a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  _QWORD *v3; // rcx
  _QWORD *v4; // rax

  v1 = a1[2];
  v2 = 0;
  if ( a1[10] != v1 )
  {
    v3 = (_QWORD *)a1[3];
    v4 = (_QWORD *)(v1 + 40);
    for ( a1[2] = v4; v3 != v4; a1[2] = v4 )
    {
      if ( *v4 != -8192 && *v4 != -4096 )
        break;
      v4 += 5;
    }
    return 1;
  }
  return v2;
}
