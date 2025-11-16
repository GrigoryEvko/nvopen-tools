// Function: sub_140B250
// Address: 0x140b250
//
__int64 __fastcall sub_140B250(_QWORD *a1)
{
  __int64 v1; // r14
  __int64 v3; // r12
  int v4; // ebx
  __int64 v5; // rdi
  __int64 v6; // rax

  v1 = 0;
  v3 = a1[1];
  v4 = 0;
  if ( !v3 )
    return *a1;
  do
  {
    while ( 1 )
    {
      v5 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      v6 = sub_1648700(v5);
      if ( *(_BYTE *)(v6 + 16) == 71 )
        break;
      if ( !v3 )
        goto LABEL_6;
    }
    v1 = *(_QWORD *)v6;
    ++v4;
  }
  while ( v3 );
LABEL_6:
  if ( v4 != 1 )
  {
    if ( v4 )
      return v3;
    return *a1;
  }
  return v1;
}
