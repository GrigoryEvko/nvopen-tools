// Function: sub_124C870
// Address: 0x124c870
//
__int64 __fastcall sub_124C870(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned int a6)
{
  int v9; // ecx
  unsigned int v10; // eax
  _QWORD *v11; // rax
  __int64 v12; // r8
  int v13; // edx
  __int16 v14; // ax
  void *v16; // rax
  char v17; // al
  __int64 v18; // [rsp-40h] [rbp-40h]
  __int64 v19; // [rsp-40h] [rbp-40h]
  char v20; // [rsp-3Fh] [rbp-3Fh]

  if ( !*a3 )
    return 0;
  v9 = *(_DWORD *)*a3 >> 8;
  if ( (unsigned __int16)v9 > 0x39u )
    goto LABEL_9;
  if ( ((1LL << v9) & 0x1C00000000010C2LL) != 0 )
    return 1;
  if ( ((1LL << v9) & 0x200000000000000LL) == 0 )
  {
LABEL_9:
    if ( *(_QWORD *)a4
      || (*(_BYTE *)(a4 + 9) & 0x70) == 0x20
      && (v19 = a5, *(char *)(a4 + 8) >= 0)
      && (*(_BYTE *)(a4 + 8) |= 8u, v16 = sub_E807D0(*(_QWORD *)(a4 + 24)), a5 = v19, (*(_QWORD *)a4 = v16) != 0) )
    {
      v18 = a5;
      if ( !sub_EA1840(a4) )
      {
        v10 = sub_EA1780(a4);
        if ( v10 > 2 )
        {
          if ( v10 != 10 )
            BUG();
          return 1;
        }
        if ( !v10 && (unsigned int)sub_EA1630(a4) != 10 )
        {
          v11 = *(_QWORD **)a4;
          v12 = v18;
          if ( !*(_QWORD *)a4
            && ((*(_BYTE *)(a4 + 9) & 0x70) != 0x20
             || *(char *)(a4 + 8) < 0
             || (*(_BYTE *)(a4 + 8) |= 8u, v11 = sub_E807D0(*(_QWORD *)(a4 + 24)), v12 = v18, (*(_QWORD *)a4 = v11) == 0))
            || off_4C5D170 == (_UNKNOWN *)v11 )
          {
LABEL_17:
            if ( !(unsigned __int8)sub_E5BBB0(a2, a4) )
              return (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD))(**(_QWORD **)(a1 + 112) + 32LL))(
                       *(_QWORD *)(a1 + 112),
                       a3,
                       a4,
                       a6);
            return 1;
          }
          v13 = *(_DWORD *)(v11[1] + 152LL);
          if ( (v13 & 0x10) != 0 )
          {
            if ( v12 )
              return 1;
            v14 = *(_WORD *)(*(_QWORD *)(a1 + 112) + 10LL);
            if ( v14 == 3 )
            {
              if ( a6 == 9 )
                return 1;
            }
            else if ( v14 == 8 )
            {
              v20 = BYTE1(v13);
              v17 = sub_124C860(a1);
              BYTE1(v13) = v20;
              if ( !v17 )
                return 1;
            }
          }
          if ( (v13 & 0x400) == 0 )
            goto LABEL_17;
        }
      }
    }
    return 1;
  }
  return 0;
}
