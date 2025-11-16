// Function: sub_2BE1280
// Address: 0x2be1280
//
__int64 __fastcall sub_2BE1280(__int64 a1)
{
  char *v1; // rax
  unsigned int v2; // r13d
  _QWORD *v3; // r12
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // r12d
  int v8; // edx
  unsigned int v9; // r14d
  _QWORD *v10; // r13
  __int64 v11; // rax
  int v12; // eax

  v1 = *(char **)(a1 + 24);
  if ( *(char **)(a1 + 32) == v1 )
  {
    v8 = *(_DWORD *)(a1 + 112);
    v6 = 0;
    if ( (v8 & 4) != 0 )
      return v6;
    if ( v1 == *(char **)(a1 + 40) )
    {
      if ( (v8 & 8) != 0 )
        return v6;
      if ( (v8 & 0x80) == 0 )
      {
LABEL_4:
        if ( v1 == *(char **)(a1 + 40) )
          return v6;
LABEL_9:
        v9 = *v1;
        v10 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL) + 80LL);
        v11 = sub_2BE10D0(v10, (__int64)"w", "", 0);
        LOBYTE(v12) = sub_2BDBFE0(v10, v9, v11, SBYTE2(v11));
        return v12 ^ v6;
      }
    }
    else if ( (v8 & 0x80) == 0 )
    {
      goto LABEL_9;
    }
LABEL_3:
    v2 = *(v1 - 1);
    v3 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL) + 80LL);
    v4 = sub_2BE10D0(v3, (__int64)"w", "", 0);
    LOBYTE(v5) = sub_2BDBFE0(v3, v2, v4, SBYTE2(v4));
    v6 = v5;
    v1 = *(char **)(a1 + 24);
    goto LABEL_4;
  }
  if ( v1 != *(char **)(a1 + 40) || (*(_BYTE *)(a1 + 112) & 8) == 0 )
    goto LABEL_3;
  return 0;
}
