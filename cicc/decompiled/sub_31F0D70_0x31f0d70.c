// Function: sub_31F0D70
// Address: 0x31f0d70
//
void (*__fastcall sub_31F0D70(__int64 a1, __int64 a2, char a3))()
{
  __int64 v4; // r14
  unsigned int v5; // eax

  if ( a3 )
    goto LABEL_4;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 259LL) )
    return (void (*)())(*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 368LL))(
                         *(_QWORD *)(a1 + 224),
                         a2,
                         0);
  if ( *(_BYTE *)(a1 + 976) )
  {
    v4 = *(_QWORD *)(a1 + 224);
    v5 = sub_31DF6B0(a1);
    return sub_E9A500(v4, a2, v5, 0);
  }
  else
  {
LABEL_4:
    sub_31DF6B0(a1);
    if ( !*(_QWORD *)a2 )
    {
      if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
        BUG();
      *(_BYTE *)(a2 + 8) |= 8u;
      *(_QWORD *)a2 = sub_E807D0(*(_QWORD *)(a2 + 24));
    }
    return (void (*)())sub_31DCA50(a1);
  }
}
