// Function: sub_18FEB70
// Address: 0x18feb70
//
__int64 __fastcall sub_18FEB70(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // edx
  int v5; // edx
  __int64 *v6; // r13
  int v7; // r12d
  __int64 v8; // r11
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // rsi
  unsigned int i; // eax
  __int64 *v15; // rbx
  __int64 v16; // r14
  unsigned int v17; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *a2;
  v10 = a2[2];
  v11 = a2[3];
  v12 = a2[4];
  v13 = a2[1];
  for ( i = v5
          & (((unsigned int)v12 >> 9)
           ^ ((unsigned int)v12 >> 4)
           ^ ((unsigned int)v11 >> 9)
           ^ ((unsigned int)v11 >> 4)
           ^ ((unsigned int)v10 >> 9)
           ^ ((unsigned int)v10 >> 4)
           ^ (37 * v13)
           ^ ((unsigned int)v9 >> 9)
           ^ ((unsigned int)v9 >> 4)); ; i = v5 & v17 )
  {
    v15 = (__int64 *)(v8 + 48LL * i);
    v16 = *v15;
    if ( v9 == *v15 && v13 == v15[1] && v10 == v15[2] && v11 == v15[3] && v12 == v15[4] )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && !v15[1] && !v15[2] && !v15[3] && !(v15[4] | (unsigned __int64)v6) )
      v6 = (__int64 *)(v8 + 48LL * i);
LABEL_7:
    v17 = v7 + i;
    ++v7;
  }
  if ( v15[1] || v15[2] || v15[3] || v15[4] )
    goto LABEL_7;
  if ( !v6 )
    v6 = (__int64 *)(v8 + 48LL * i);
  *a3 = v6;
  return 0;
}
