// Function: sub_130BF80
// Address: 0x130bf80
//
__int64 __fastcall sub_130BF80(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 *a4)
{
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  char v10; // cl
  unsigned int v11; // eax
  unsigned int v12; // edx
  __int64 result; // rax
  int v14; // eax
  unsigned int v15; // edx
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  if ( !a4 )
  {
    v12 = 0;
LABEL_8:
    v16 = v12;
    v14 = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 58472));
    v15 = v16;
    if ( v14 )
    {
      sub_130AD90(a2 + 58408);
      *(_BYTE *)(a2 + 58512) = 1;
      v15 = v16;
    }
    ++*(_QWORD *)(a2 + 58464);
    if ( a1 != *(_QWORD *)(a2 + 58456) )
    {
      ++*(_QWORD *)(a2 + 58448);
      *(_QWORD *)(a2 + 58456) = a1;
    }
    if ( a3 )
      *a3 = qword_5060180[*(unsigned int *)(a2 + 58404)];
    if ( a4 )
      *(_DWORD *)(a2 + 58404) = v15;
    *(_BYTE *)(a2 + 58512) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 58472));
    return 0;
  }
  v6 = *a4;
  v7 = *a4 + 1;
  if ( v7 > 0x7000000000000000LL )
  {
    v12 = 198;
    goto LABEL_8;
  }
  _BitScanReverse64(&v8, v7);
  v9 = v8 - (((v6 & v7) == 0) - 1);
  if ( v9 < 0xE )
    v9 = 14;
  v10 = v9 - 3;
  v11 = v9 - 14;
  if ( !v11 )
    v10 = 12;
  v12 = ((v6 >> v10) & 3) + 4 * v11 - 1;
  result = 1;
  if ( v12 <= 0xC6 )
    goto LABEL_8;
  return result;
}
