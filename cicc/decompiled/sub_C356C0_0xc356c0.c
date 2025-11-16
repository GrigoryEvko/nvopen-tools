// Function: sub_C356C0
// Address: 0xc356c0
//
__int64 __fastcall sub_C356C0(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // cl
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdx

  v2 = *(_BYTE *)(a2 + 20);
  v3 = v2 & 7;
  if ( (v2 & 7) == 1 )
    goto LABEL_10;
  if ( v3 == 3 )
  {
    LOBYTE(v4) = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
    v6 = 0;
    goto LABEL_8;
  }
  if ( !v3 )
LABEL_10:
    BUG();
  v4 = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 2;
  v5 = *(_QWORD *)sub_C33930(a2);
  if ( v4 == 1 )
  {
    v4 = *(_QWORD *)sub_C33930(a2);
    v6 = v5 & 3;
    v2 = *(_BYTE *)(a2 + 20);
    LOBYTE(v4) = (v4 & 4) != 0;
  }
  else
  {
    v2 = *(_BYTE *)(a2 + 20);
    v6 = v5 & 3;
  }
LABEL_8:
  *(_DWORD *)(a1 + 8) = 6;
  *(_QWORD *)a1 = (4 * (_BYTE)v4) & 0x1C | v6 | (32LL * ((v2 & 8) != 0));
  return a1;
}
