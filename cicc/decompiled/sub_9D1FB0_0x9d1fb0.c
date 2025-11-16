// Function: sub_9D1FB0
// Address: 0x9d1fb0
//
unsigned __int64 __fastcall sub_9D1FB0(__int64 *a1, __int64 a2, __int16 *a3, const void *a4, __int64 a5)
{
  size_t v5; // r15
  __int64 v7; // rsi
  int v8; // r13d
  __int64 v10; // rax
  unsigned __int64 v11; // r12
  __int16 v12; // ax
  __int64 v13; // rax
  unsigned int v15; // eax
  unsigned int v16; // eax

  v5 = 4 * a5;
  v7 = 4 * a5 + 88;
  v8 = a5;
  v10 = *a1;
  a1[10] += v7;
  v11 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[1] >= v7 + v11 && v10 )
    *a1 = v7 + v11;
  else
    v11 = sub_9D1E70((__int64)a1, v7, 4 * a5 + 88, 3);
  sub_BD35F0(v11, a2, 255);
  v12 = *a3;
  *(_DWORD *)(v11 + 28) = v8;
  *(_WORD *)(v11 + 24) = v12;
  *(_DWORD *)(v11 + 32) = *((_DWORD *)a3 + 1);
  v13 = *((_QWORD *)a3 + 1);
  *(_BYTE *)(v11 + 80) = 0;
  *(_QWORD *)(v11 + 40) = v13;
  if ( *((_BYTE *)a3 + 48) )
  {
    v15 = *((_DWORD *)a3 + 6);
    *(_DWORD *)(v11 + 56) = v15;
    if ( v15 > 0x40 )
      sub_C43780(v11 + 48, a3 + 8);
    else
      *(_QWORD *)(v11 + 48) = *((_QWORD *)a3 + 2);
    v16 = *((_DWORD *)a3 + 10);
    *(_DWORD *)(v11 + 72) = v16;
    if ( v16 > 0x40 )
      sub_C43780(v11 + 64, a3 + 16);
    else
      *(_QWORD *)(v11 + 64) = *((_QWORD *)a3 + 4);
    *(_BYTE *)(v11 + 80) = 1;
  }
  if ( v5 )
    memmove((void *)(v11 + 88), a4, v5);
  return v11;
}
