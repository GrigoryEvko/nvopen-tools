// Function: sub_397C410
// Address: 0x397c410
//
void (*__fastcall sub_397C410(__int64 a1, __int64 a2, char a3))()
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax

  if ( a3 )
    goto LABEL_13;
  v3 = *(_QWORD *)(a1 + 240);
  if ( *(_BYTE *)(v3 + 282) )
    return (void (*)())(*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 328LL))(
                         *(_QWORD *)(a1 + 256),
                         a2,
                         0);
  if ( *(_BYTE *)(v3 + 356) )
    return sub_38DDC80(*(__int64 **)(a1 + 256), a2, 4u, 0);
LABEL_13:
  if ( (*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8
      || (*(_BYTE *)(a2 + 8) |= 4u,
          v4 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24)),
          *(_QWORD *)a2 = v4 | *(_QWORD *)a2 & 7LL,
          !v4) )
    {
      BUG();
    }
  }
  return (void (*)())sub_396F380(a1);
}
