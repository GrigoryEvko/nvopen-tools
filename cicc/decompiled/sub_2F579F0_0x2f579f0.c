// Function: sub_2F579F0
// Address: 0x2f579f0
//
__int64 __fastcall sub_2F579F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int *a5, char a6)
{
  int v6; // eax
  int v7; // r15d
  int v8; // r13d
  unsigned int v11; // esi
  int v12; // r11d
  __int64 v13; // r10
  __int64 v14; // r9
  unsigned __int16 *v15; // rsi
  __int64 v16; // r9
  int v17; // r11d
  int v21; // [rsp+1Ch] [rbp-44h]
  unsigned int v22; // [rsp+28h] [rbp-38h] BYREF
  int v23[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = *(_DWORD *)(a3 + 72);
  v7 = -*(_DWORD *)(a3 + 8);
  v22 = -1;
  v21 = v6;
  if ( v6 == v7 )
    return 0xFFFFFFFFLL;
  v8 = v7;
  do
  {
LABEL_4:
    if ( v8 < 0 )
      v11 = *(unsigned __int16 *)(*(_QWORD *)a3 + 2 * (*(_QWORD *)(a3 + 8) + v8));
    else
      v11 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 56) + 2LL * v8);
    if ( !a6 || !(unsigned __int8)sub_2F50F60(*(_QWORD *)(a1 + 968), v11) )
      sub_2F57410(a1, v11, a3, a4, a5, &v22);
    v12 = *(_DWORD *)(a3 + 72);
    if ( v12 > v8 && ++v8 >= 0 && v12 > v8 )
    {
      v13 = *(_QWORD *)(a3 + 56);
      v14 = v8;
      while ( 1 )
      {
        v8 = v14;
        if ( (unsigned int)*(unsigned __int16 *)(v13 + 2 * v14) - 1 > 0x3FFFFFFE )
          break;
        v23[0] = *(unsigned __int16 *)(v13 + 2 * v14);
        v15 = (unsigned __int16 *)(*(_QWORD *)a3 + 2LL * *(_QWORD *)(a3 + 8));
        if ( v15 == sub_2F4C810(*(unsigned __int16 **)a3, (__int64)v15, v23) )
          break;
        v14 = v16 + 1;
        ++v8;
        if ( v17 <= (int)v14 )
        {
          if ( v21 != v8 )
            goto LABEL_4;
          return v22;
        }
      }
    }
  }
  while ( v21 != v8 );
  return v22;
}
