// Function: sub_68FE10
// Address: 0x68fe10
//
__int64 __fastcall sub_68FE10(_BYTE *a1, int a2, int a3)
{
  __int64 v5; // rdi
  __int64 result; // rax
  char v7; // si
  char v8; // cl
  __int64 v9; // rdx
  __int64 v10; // rdx

  v5 = *(_QWORD *)a1;
  if ( a2 )
    result = sub_8E3200(v5);
  else
    result = sub_8E3180(v5);
  if ( (_DWORD)result )
    return 1;
  v7 = a1[16];
  if ( !v7 )
    return 1;
  v8 = *(_BYTE *)(*(_QWORD *)a1 + 140LL);
  if ( v8 == 12 )
  {
    v9 = *(_QWORD *)a1;
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
      v8 = *(_BYTE *)(v9 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 || v7 == 5 )
    return 1;
  if ( !a3 )
    return 0;
  if ( dword_4F04C44 != -1 )
    return (unsigned int)sub_8DBE70(*(_QWORD *)a1) != 0;
  v10 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v10 + 6) & 6) != 0 || *(_BYTE *)(v10 + 4) == 12 )
    return (unsigned int)sub_8DBE70(*(_QWORD *)a1) != 0;
  return result;
}
