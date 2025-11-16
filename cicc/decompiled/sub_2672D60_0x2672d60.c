// Function: sub_2672D60
// Address: 0x2672d60
//
__int64 __fastcall sub_2672D60(__int64 a1, __int64 a2, unsigned __int8 *a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  int v8; // edx
  int v9; // edi
  int v10; // edi
  int v11; // edx
  int v12; // ecx
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // r13d
  __int64 v18; // rsi
  unsigned int v19; // eax
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rdx

  v8 = *a3;
  if ( (_BYTE)a5 )
  {
    v9 = v8;
    LOBYTE(v8) = a5;
    v10 = v9 ^ 1;
  }
  else
  {
    v10 = 0;
    if ( !*a4 )
    {
      v10 = v8;
      LOBYTE(v8) = 0;
    }
  }
  *a3 = v8;
  v11 = a3[3];
  v12 = a3[1];
  v13 = v11 ^ 1;
  if ( (_BYTE)v12 && a4[1] )
  {
    a3[1] = 1;
    v14 = (unsigned __int8)a4[3];
    v15 = (unsigned int)v14 | v11;
    a3[3] = v15;
    v16 = v10 | v14 & v13;
    sub_2672B10((__int64)a3, (__int64)a4, v15, v14, a5, a6);
    return v16;
  }
  a3[1] = 0;
  v18 = (unsigned __int8)a4[3];
  ++*((_QWORD *)a3 + 7);
  a3[3] = v18 | v11;
  v16 = v10 | v12 | v18 & v13;
  if ( a3[84] )
    goto LABEL_8;
  v21 = 4 * (*((_DWORD *)a3 + 19) - *((_DWORD *)a3 + 20));
  v22 = *((unsigned int *)a3 + 18);
  if ( v21 < 0x20 )
    v21 = 32;
  if ( (unsigned int)v22 <= v21 )
  {
    v18 = 0xFFFFFFFFLL;
    memset(*((void **)a3 + 8), -1, 8 * v22);
LABEL_8:
    *(_QWORD *)(a3 + 76) = 0;
    goto LABEL_9;
  }
  sub_C8C990((__int64)(a3 + 56), v18);
LABEL_9:
  ++*((_QWORD *)a3 + 1);
  if ( !a3[36] )
  {
    v19 = 4 * (*((_DWORD *)a3 + 7) - *((_DWORD *)a3 + 8));
    v20 = *((unsigned int *)a3 + 6);
    if ( v19 < 0x20 )
      v19 = 32;
    if ( (unsigned int)v20 > v19 )
    {
      sub_C8C990((__int64)(a3 + 8), v18);
      return v16;
    }
    memset(*((void **)a3 + 2), -1, 8 * v20);
  }
  *(_QWORD *)(a3 + 28) = 0;
  return v16;
}
