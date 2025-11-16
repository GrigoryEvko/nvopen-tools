// Function: sub_1B42200
// Address: 0x1b42200
//
__int64 __fastcall sub_1B42200(__int64 *a1, __int64 *a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rsi
  __int64 result; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx

  v3 = sub_1648700((__int64)a2);
  if ( *((_BYTE *)v3 + 16) == 77 )
  {
    if ( (*((_BYTE *)v3 + 23) & 0x40) != 0 )
      v4 = (_QWORD *)*(v3 - 1);
    else
      v4 = &v3[-3 * (*((_DWORD *)v3 + 5) & 0xFFFFFFF)];
    result = sub_1B3FB80(a1, v4[3 * *((unsigned int *)v3 + 14) + 1 + -1431655765 * (unsigned int)(a2 - v4)]);
  }
  else
  {
    result = sub_1B3FB80(a1, v3[5]);
  }
  if ( *a2 )
  {
    v6 = a2[1];
    v7 = a2[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *a2 = result;
  if ( result )
  {
    v8 = *(_QWORD *)(result + 8);
    a2[1] = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(v8 + 16) & 3LL;
    a2[2] = (result + 8) | a2[2] & 3;
    *(_QWORD *)(result + 8) = a2;
  }
  return result;
}
