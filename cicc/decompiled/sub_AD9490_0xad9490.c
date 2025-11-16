// Function: sub_AD9490
// Address: 0xad9490
//
unsigned __int8 *__fastcall sub_AD9490(unsigned __int8 *a1, __int64 a2, char a3, unsigned __int8 a4)
{
  int v4; // eax
  __int64 v6; // rax

  v4 = *a1;
  if ( (unsigned int)(v4 - 42) <= 0x11 )
    return sub_AD93D0(v4 - 29, a2, a3, a4);
  if ( (_BYTE)v4 == 85 )
  {
    v6 = *((_QWORD *)a1 - 4);
    if ( v6 )
    {
      if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
        return (unsigned __int8 *)sub_AD66B0(*(_DWORD *)(v6 + 36), a2);
    }
  }
  return 0;
}
