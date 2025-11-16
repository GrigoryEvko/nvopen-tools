// Function: sub_2473720
// Address: 0x2473720
//
__int64 __fastcall sub_2473720(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rdi
  int v9; // edx
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax

  if ( (_BYTE)qword_4FE8308 )
    sub_2463100((unsigned __int8 *)a2);
  v4 = 0;
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v6 = *(_QWORD *)(a2 - 8);
      else
        v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v7 = *(_QWORD *)(v6 + 32 * v4);
      v8 = *(_QWORD *)(v7 + 8);
      v9 = *(unsigned __int8 *)(v8 + 8);
      if ( (_BYTE)v9 == 12 || (unsigned __int8)v9 <= 3u || (_BYTE)v9 == 5 || (v9 & 0xFB) == 0xA || (v9 & 0xFD) == 4 )
        goto LABEL_8;
      if ( (unsigned __int8)(*(_BYTE *)(v8 + 8) - 15) > 3u && v9 != 20 )
        goto LABEL_9;
      if ( (unsigned __int8)sub_BCEBA0(v8, 0) )
      {
LABEL_8:
        sub_2472230((__int64)a1, v7, a2);
LABEL_9:
        if ( v5 == ++v4 )
          break;
      }
      else if ( v5 == ++v4 )
      {
        break;
      }
    }
  }
  v10 = *(_QWORD *)(a2 + 8);
  v11 = sub_2463540(a1, v10);
  v12 = (__int64)v11;
  if ( v11 )
    v12 = sub_AD6530((__int64)v11, v10);
  sub_246EF60((__int64)a1, a2, v12);
  v13 = sub_AD6530(*(_QWORD *)(a1[1] + 88), a2);
  return sub_246F1C0((__int64)a1, a2, v13);
}
