// Function: sub_F388A0
// Address: 0xf388a0
//
__int64 __fastcall sub_F388A0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD *i; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rsi
  __int64 v14; // rcx
  __int64 result; // rax
  _QWORD *v16; // rdx

  v8 = 0;
  v9 = *a2;
  for ( i = (_QWORD *)*a3; i != (_QWORD *)v9; v9 = (v11 + 8) | 4 )
  {
    while ( 1 )
    {
      v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v9 & 4) != 0 )
        break;
      ++v8;
      v9 = v11 + 144;
      if ( i == (_QWORD *)(v11 + 144) )
        goto LABEL_6;
    }
    ++v8;
  }
LABEL_6:
  v12 = *(unsigned int *)(a1 + 8);
  if ( v12 + v8 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12 + v8, 8u, v12 + v8, a6);
    v12 = *(unsigned int *)(a1 + 8);
  }
  v13 = (_QWORD *)*a3;
  v14 = *(_QWORD *)a1 + 8 * v12;
  result = *a2;
  if ( *a3 != *a2 )
  {
    do
    {
      while ( 1 )
      {
        v14 += 8;
        v16 = (_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
        if ( (result & 4) != 0 )
          break;
        *(_QWORD *)(v14 - 8) = v16[17];
        result = (__int64)(v16 + 18);
        if ( v13 == v16 + 18 )
          goto LABEL_13;
      }
      *(_QWORD *)(v14 - 8) = *(_QWORD *)(*v16 + 136LL);
      result = (unsigned __int64)(v16 + 1) | 4;
    }
    while ( v13 != (_QWORD *)result );
LABEL_13:
    LODWORD(v12) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v8 + v12;
  return result;
}
