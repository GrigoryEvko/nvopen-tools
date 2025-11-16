// Function: sub_FC7C70
// Address: 0xfc7c70
//
__int64 __fastcall sub_FC7C70(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  _QWORD *v9; // rcx
  __int64 i; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 result; // rax
  _QWORD *v16; // rdx

  v7 = 0;
  v9 = (_QWORD *)*a3;
  for ( i = *a2; v9 != (_QWORD *)i; i = (v11 + 8) | 4 )
  {
    while ( 1 )
    {
      v11 = i & 0xFFFFFFFFFFFFFFF8LL;
      if ( (i & 4) != 0 || !v11 )
        break;
      ++v7;
      i = v11 + 144;
      if ( v9 == (_QWORD *)(v11 + 144) )
        goto LABEL_7;
    }
    ++v7;
  }
LABEL_7:
  v12 = *(unsigned int *)(a1 + 8);
  if ( v12 + v7 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12 + v7, 8u, v12 + v7, a6);
    v12 = *(unsigned int *)(a1 + 8);
  }
  v13 = (_QWORD *)*a3;
  v14 = *(_QWORD *)a1 + 8 * v12;
  result = *a2;
  if ( *a3 != *a2 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 += 8;
        v16 = (_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
        if ( (result & 4) == 0 )
          break;
        *(_QWORD *)(v14 - 8) = *(_QWORD *)(*v16 + 136LL);
LABEL_12:
        result = (unsigned __int64)(v16 + 1) | 4;
        if ( v13 == (_QWORD *)result )
          goto LABEL_16;
      }
      *(_QWORD *)(v14 - 8) = v16[17];
      result = (__int64)(v16 + 18);
      if ( !v16 )
        goto LABEL_12;
      if ( v13 == v16 + 18 )
      {
LABEL_16:
        LODWORD(v12) = *(_DWORD *)(a1 + 8);
        break;
      }
    }
  }
  *(_DWORD *)(a1 + 8) = v7 + v12;
  return result;
}
