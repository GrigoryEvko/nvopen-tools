// Function: sub_9377A0
// Address: 0x9377a0
//
__int64 __fastcall sub_9377A0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5, _QWORD *a6)
{
  int v10; // eax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  char v13; // dl
  int v14; // edx
  __int64 v15; // rsi
  __int64 result; // rax
  __int64 v17; // rdx
  char v18; // cl
  unsigned int v19; // edx
  __int64 v20; // [rsp+10h] [rbp-40h]

  *(_QWORD *)a1 = 0;
  v10 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 8) = v10;
  v20 = (unsigned int)(v10 + 1);
  v11 = sub_2207820(40 * v20);
  if ( v11 && v20 )
  {
    v12 = (_QWORD *)v11;
    do
    {
      *v12 = 0;
      v12 += 5;
      *((_DWORD *)v12 - 8) = 0;
      *((_DWORD *)v12 - 7) = 0;
      *((_BYTE *)v12 - 24) = 0;
    }
    while ( v12 != (_QWORD *)(v11 + 40 * v20) );
  }
  v13 = *(_BYTE *)(a2 + 140);
  *(_QWORD *)(a1 + 16) = v11;
  *(_QWORD *)(v11 + 24) = a2;
  if ( (v13 & 0xFB) == 8 )
  {
    v19 = sub_8D4C10(a2, dword_4F077C4 != 2);
    v11 = *(_QWORD *)(a1 + 16);
    v14 = (v19 >> 2) & 1;
  }
  else
  {
    LOBYTE(v14) = 0;
  }
  v15 = *(unsigned int *)(a1 + 8);
  *(_BYTE *)(v11 + 32) = v14;
  result = v11 + 64;
  *(_WORD *)(result - 31) = 0;
  v17 = 0;
  if ( (_DWORD)v15 )
  {
    do
    {
      result += 40;
      *(_QWORD *)(result - 40) = *(_QWORD *)(*(_QWORD *)a3 + 8 * v17);
      *(_BYTE *)(result - 32) = *(_BYTE *)(*a4 + v17);
      *(_BYTE *)(result - 31) = *(_BYTE *)(*a5 + v17);
      v18 = *(_BYTE *)(*a6 + v17++);
      *(_BYTE *)(result - 30) = v18;
    }
    while ( v17 != v15 );
  }
  return result;
}
