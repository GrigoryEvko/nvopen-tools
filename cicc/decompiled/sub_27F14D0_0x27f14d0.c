// Function: sub_27F14D0
// Address: 0x27f14d0
//
void __fastcall sub_27F14D0(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 *a8)
{
  unsigned __int64 v11; // rsi
  int v12; // eax
  unsigned __int64 v13; // rsi
  char v14; // dh
  __int64 v15; // rsi
  char v16; // dl
  __int64 v17; // rax

  sub_27F11B0(a8, a1);
  if ( (a1[7] & 0x20) != 0 || *a1 == 85 )
  {
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 *, __int64, __int64))(*(_QWORD *)a5 + 24LL))(
            a5,
            a1,
            a2,
            a3) )
      sub_B44E20(a1);
    if ( *a1 != 84 )
      goto LABEL_4;
  }
  else if ( *a1 != 84 )
  {
LABEL_4:
    v11 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == a4 + 48 )
    {
      v13 = 0;
    }
    else
    {
      if ( !v11 )
        BUG();
      v12 = *(unsigned __int8 *)(v11 - 24);
      v13 = v11 - 24;
      if ( (unsigned int)(v12 - 30) >= 0xB )
        v13 = 0;
    }
    sub_27EC7D0(a1, v13 + 24, 0, a5, a6, a7);
    goto LABEL_9;
  }
  v15 = sub_AA4FF0(a4);
  v16 = 0;
  if ( v15 )
    v16 = v14;
  v17 = 1;
  BYTE1(v17) = v16;
  sub_27EC7D0(a1, v15, v17, a5, a6, a7);
LABEL_9:
  sub_AE9120((char *)a1);
}
