// Function: sub_C44D10
// Address: 0xc44d10
//
void __fastcall sub_C44D10(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 **v2; // r13
  unsigned __int64 v3; // r15
  unsigned int v4; // r14d
  unsigned int v5; // ebx
  __int128 v6; // rax
  unsigned __int64 v8; // rax
  int v9; // eax

  v2 = (unsigned __int64 **)a2;
  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_DWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a1 + 8);
  if ( v4 > 0x40 )
  {
    v9 = sub_C444A0(a2);
    LODWORD(a2) = v3;
    if ( v4 - v9 <= 0x40 )
    {
      a2 = **v2;
      if ( v3 < a2 )
        LODWORD(a2) = v3;
      if ( v5 <= 0x40 )
        goto LABEL_5;
LABEL_16:
      sub_C44B70(a1, a2);
      return;
    }
  }
  else
  {
    a2 = *(_QWORD *)a2;
    if ( v3 < a2 )
      LODWORD(a2) = *(_DWORD *)(a1 + 8);
  }
  if ( v5 > 0x40 )
    goto LABEL_16;
LABEL_5:
  *(_QWORD *)&v6 = 0;
  if ( v5 )
    *(_QWORD *)&v6 = (__int64)(*(_QWORD *)a1 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
  v6 = (__int64)v6;
  *(_QWORD *)&v6 = (__int64)v6 >> a2;
  if ( v5 != (_DWORD)a2 )
    *((_QWORD *)&v6 + 1) = v6;
  v8 = *((_QWORD *)&v6 + 1) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v5);
  if ( !v5 )
    v8 = 0;
  *(_QWORD *)a1 = v8;
}
