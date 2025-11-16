// Function: sub_373B120
// Address: 0x373b120
//
__int64 __fastcall sub_373B120(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // r12

  v3 = a1[11];
  a1[21] += 48;
  v5 = (v3 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( a1[12] >= (unsigned __int64)(v5 + 48) && v3 )
  {
    a1[11] = v5 + 48;
    if ( !v5 )
      goto LABEL_5;
  }
  else
  {
    v5 = sub_9D1E70((__int64)(a1 + 11), 48, 48, 4);
  }
  *(_BYTE *)(v5 + 30) = 0;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)v5 = v5 | 4;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)(v5 + 24) = -1;
  *(_WORD *)(v5 + 28) = 10;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = 0;
LABEL_5:
  sub_324C3F0((__int64)a1, *(unsigned __int8 **)(a2 + 8), v5);
  *(_QWORD *)(a2 + 24) = v5;
  if ( *(_BYTE *)(a3 + 24) )
    sub_3736500(a1, a2, v5);
  return v5;
}
